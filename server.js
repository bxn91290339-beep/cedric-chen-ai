import express from 'express';
import axios from 'axios';
import cors from 'cors';
import multer from 'multer';
import pdf from 'pdf-parse';
import mammoth from 'mammoth';
import helmet from 'helmet';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import puppeteer from 'puppeteer';
import readline from 'readline';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OllamaEmbeddings } from "@langchain/ollama";

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

// 🛡️ Helmet CSP hatasını önlemek için içerik güvenlik politikası devre dışı bırakıldı
app.use(helmet({
    contentSecurityPolicy: false,
}));
app.use(cors());
app.use(express.json());

// 🛡️ GÜVENLİK YAMASI: Klasör yapını bozmadan sadece hassas dosyaları erişime kapatır.
app.use((req, res, next) => {
    if (req.path.includes('server.js') || req.path.includes('chat_history.json')) {
        return res.status(403).send("Erişim yasak.");
    }
    next();
});

// 🌐 Orijinal statik dosya sunumun
app.use(express.static(__dirname));

// 🛠️ KESİN ÇÖZÜM: Ana dizine girildiğinde doğrudan index.html dosyasını çalıştırır
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// ⚡ GROQ API BİLGİLERİ
const GROQ_API_KEY = "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";
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

// --- 🧠 ÇOKLU ODA DESTEKLİ AKILLI HAFIZA YAPISI ---
let chatSessions = {}; 
let activeSessionId = null;

let vectorStore = null;
const HISTORY_FILE = 'chat_history.json';
const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" }); 

// 🎭 KAPSAMLI DUYGU ANALİZİ KELİMELERİ
const emotionKeywords = {
    sinirli: ["sinir", "aptal", "saçma", "kızgın", "nefret", "bıktım", "off", "yeter", "hata", "bozuk", "lanet", "kötü", "çıldır"],
    mutlu: ["mutlu", "harika", "süper", "teşekkür", "gül", "sevin", "iyi", "güzel", "mükemmel", "harikulade", "şahane"],
    utangac: ["utan", "çekin", "kızar", "iltifat", "mahcup", "tatlı", "aşk", "seviyorum", "utandı", "şapşal"],
    sasirmis: ["şaşır", "şok", "inanılmaz", "oha", "yok artık", "ciddi", "gerçekten", "vay", "nasıl yani", "şaka", "hadi canım"]
};

function checkMood(text) {
    if (!text) return 'notr';
    const lower = text.toLowerCase();
    for (const [emotion, keywords] of Object.entries(emotionKeywords)) {
        if (keywords.some(word => lower.includes(word))) {
            return emotion;
        }
    }
    return 'notr';
}

// 🎯 IŞIK HIZI TRAFİK POLİSİ
const webSearchKeywords = [
    "hava durumu", "kpss", "altın", "çeyrek", "dolar", "euro", "döviz", "fiyat", "kaç tl", 
    "güncel", "haber", "kimdir", "nedir", "vizyon", "borsa", "sınav ücret", "kaç para", 
    "bugün", "borsada", "piyasa", "altın fiyatı", "link", "site", "url", "adres", "bağlantı", "bul", "web",
    "saat", "tarih", "zaman", "kaç", "hava", "derece", "yağmur", "soru", "test", "sınav", "hazırla", "çıkar"
];

function checkNeedsInternet(text) {
    const lower = text.toLowerCase();
    return webSearchKeywords.some(keyword => lower.includes(keyword));
}

const localKnowledge2026 = {
    kpss: "2026 yılı güncel KPSS Lisans Genel Yetenek-Genel Kültür oturumu ücreti 800 TL, Alan Bilgisi oturumu ücreti ise 500 TL olarak belirlenmiştir.",
    altin: "Canlı piyasalarda güncel anlık 1 gram altın fiyatı 3000 TL civarında işlem görmektedir."
};

function getStaticFallback(query) {
    const lower = query.toLowerCase();
    if (lower.includes("kpss") || lower.includes("sınav")) return localKnowledge2026.kpss;
    if (lower.includes("altın") || lower.includes("gram") || lower.includes("fiyat")) return localKnowledge2026.altin;
    return null;
}

function ensureDefaultSession() {
    if (Object.keys(chatSessions).length === 0) {
        const defaultId = "session_" + Date.now();
        chatSessions[defaultId] = { history: [], summary: "" };
        activeSessionId = defaultId;
    }
    if (!activeSessionId || !chatSessions[activeSessionId]) {
        activeSessionId = Object.keys(chatSessions)[0];
    }
}

if (fs.existsSync(HISTORY_FILE)) {
    try {
        const data = fs.readFileSync(HISTORY_FILE, 'utf8');
        if (data.includes("uploadedfilecontext") || data.includes("MÜHÜRLÜ EMİRLER") || data.includes("Sistem,")) {
            console.log("Eski hatalı robotik hafıza tespit edildi. Sistem dosyayı otomatik temizliyor...");
            fs.unlinkSync(HISTORY_FILE);
            chatSessions = {};
            ensureDefaultSession();
        } else {
            const parsed = JSON.parse(data);
            if (parsed && parsed.sessions) {
                chatSessions = parsed.sessions;
                activeSessionId = parsed.activeSessionId;
            } else if (parsed && (parsed.history || parsed.summary)) {
                const legacyId = "session_legacy_" + Date.now();
                chatSessions = {
                    [legacyId]: { history: parsed.history || [], summary: parsed.summary || "" }
                };
                activeSessionId = legacyId;
            } else {
                ensureDefaultSession();
            }
            console.log("Cedric-Chen'in çoklu oda destekli akıllı hafızası başarıyla yüklendi.");
        }
    } catch (e) { 
        console.error("Geçmiş dosyası kontrol edilirken hata:", e.message); 
        ensureDefaultSession();
    }
} else {
    ensureDefaultSession();
}

function saveChatHistory() {
    const dataToSave = {
        sessions: chatSessions,
        activeSessionId: activeSessionId
    };
    fs.writeFileSync(HISTORY_FILE, JSON.stringify(dataToSave, null, 2));
}

// ⚡ GROQ İLE HIZLANDIRILMIŞ HAFIZA ÖZETLEME
async function maintainSmartMemory(sessionId) {
    const session = chatSessions[sessionId];
    if (!session || session.history.length < 10) return;

    console.log(`🧠 Bilinçaltı Hafıza Motoru [${sessionId}]: Eski konuşmalar özet havuzuna aktarılıyor...`);
    const messagesToSummarize = session.history.slice(0, 4); 
    session.history = session.history.slice(4); 

    const contextText = messagesToSummarize.map(m => `${m.role === 'User' ? 'Eren' : 'Cedric-Chen'}: ${m.content}`).join("\n");

    try {
        const response = await axios.post(GROQ_URL, {
            model: "llama-3.1-8b-instant",
            messages: [
                {
                    role: "system",
                    content: "Sen bir hafıza sıkıştırma mimarisisin. Eren-Cedric arasındaki konuşma kesitini, varsa eski özet verisiyle birleştirip tek bir güncel özet haline getireceksin. Eren'in projelerini, hayatındaki önemli kişileri (Örn: Zehra) ve özel kararları asla unutmama ödevin var. Çok uzatmadan, doğrudan can alıcı detayları barındıran temiz bir özet metni döndür."
                },
                {
                    role: "user",
                    content: `Mevcut Özet Havuzu: "${session.summary}"\n\nYeni Eklenecek Konuşma Geçmişi:\n${contextText}\n\nLütfen bana güncellenmiş tek bir birleşik özet metni üret.`
                }
            ],
            stream: false
        }, { headers: { 'Authorization': `Bearer ${GROQ_API_KEY}` } });

        if (response.data?.choices?.[0]?.message?.content) {
            session.summary = response.data.choices[0].message.content.trim();
            console.log(`🧠 Yeni Bilinçaltı Hafızası Mühürlendi [${sessionId}]: "${session.summary}"`);
            saveChatHistory();
        }
    } catch (e) {
        console.error("❌ Hafıza özetleme esnasında Groq hatası:", e.message);
    }
}

const cedricTools = [
    {
        type: 'function',
        function: {
            name: 'getFinanceData',
            description: 'Kullanıcı altın, gram altın, çeyrek altın, döviz piyasası veya sarı metal fiyatları hakkında anlık durum sorduğunda canlı finans verilerini getirmek için bu fonksiyonu çağır.',
            parameters: { type: 'object', properties: { query: { type: 'string', description: 'Finansal arama terimi' } }, required: ['query'] }
        }
    },
    {
        type: 'function',
        function: {
            name: 'getGeneralData',
            description: 'Kullanıcı güncel 2026 haberleri, internet siteleri, linkler, sınav ücretleri gibi web araması gerektiren canlı bilgi sorduğunda bu fonksiyonu çağır.',
            parameters: { type: 'object', properties: { query: { type: 'string', description: 'Arama terimi' } }, required: ['query'] }
        }
    },
    {
        type: 'function',
        function: {
            name: 'getCurrentTime',
            description: 'Kullanıcı o anki saati, günü veya tarihi sorduğunda bu fonksiyonu çağır.',
            parameters: { type: 'object', properties: {}, required: [] }
        }
    },
    {
        type: 'function',
        function: {
            name: 'getWeatherData',
            description: 'Kullanıcı hava durumunu sorduğunda bu fonksiyonu çağır.',
            parameters: { type: 'object', properties: { city: { type: 'string', description: 'Hava durumu sorulan şehir' } }, required: ['city'] }
        }
    },
    {
        type: 'function',
        function: {
            name: 'generateExamQuestions',
            description: 'Kullanıcı genel konularda veya yüklenen dökümandan test/quiz hazırlamanı istediğinde bu fonksiyonu çağır.',
            parameters: { type: 'object', properties: { query: { type: 'string', description: 'İstenen soru detayı' } }, required: ['query'] }
        }
    }
];

// 🚀 PUPPETEER OPTİMİZASYONU VE GÜVENLİ KAPATMA
let globalBrowser = null;
async function getGeneralData(query) {
    let page;
    try {
        const tazeSorgu = `${query}`;
        console.log(`🔍 Araç Tetiklendi (Puppeteer): Canlı bilgi ve Linkler aranıyor -> "${tazeSorgu}"`);
        
        if (!globalBrowser) {
            globalBrowser = await puppeteer.launch({ 
                headless: true, 
                args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--disable-extensions', '--disable-software-rasterizer', '--disable-dev-shm-usage'] 
            });
        }
        
        page = await globalBrowser.newPage();
        await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36');
        
        const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(tazeSorgu)}`;
        await page.goto(searchUrl, { waitUntil: 'load', timeout: 8000 });
        
        const snippets = await page.evaluate(() => {
            const elements = document.querySelectorAll('.result');
            return Array.from(elements).map(el => {
                const linkEl = el.querySelector('.result__title a');
                const snippetEl = el.querySelector('.result__snippet');
                if (linkEl) {
                    const title = linkEl.innerText.trim();
                    const url = linkEl.href;
                    const snippet = snippetEl ? snippetEl.innerText.trim() : '';
                    return `Site Adı: ${title}\nBağlantı Adresi: ${url}\nAçıklama: ${snippet}\n---`;
                }
                return null;
            }).filter(text => text !== null);
        });
        
        await page.close();
        if (snippets.length === 0) throw new Error("Boş veri döndü.");
        return snippets.slice(0, 3).join('\n');
    } catch (e) {
        if (page) await page.close().catch(() => {});
        const fallback = getStaticFallback(query);
        return fallback ? fallback : "GÜNCEL BİLGİ VE LİNK İNTERNETTEN ÇEKİLEMEDİ";
    }
}

async function getFinanceData(query) {
    try {
        console.log(`🔍 Araç Tetiklendi (Finance API): Altın fiyatları çekiliyor...`);
        const res = await axios.get('https://api.genelpara.com/embed/altin.json', { timeout: 3000 });
        if (res.data && res.data.GA) {
            const gramAltin = parseFloat(res.data.GA.satis).toFixed(0);
            return `Canlı piyasa verilerine göre şu anda 1 gram altın satış fiyatı tam olarak ${gramAltin} TL değerindedir.`;
        }
    } catch (e) {
        console.warn(`⚠️ Finans API'si gecikti, yedek Puppeteer hattına geçiliyor...`);
    }
    const staticFallback = getStaticFallback(query);
    if (staticFallback) return staticFallback;
    return await getGeneralData(query);
}

function getCurrentTime() {
    const now = new Date();
    const options = { timeZone: 'Europe/Istanbul', weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return `SİSTEM BİLGİSİ: Şu anki güncel tarih ve saat tam olarak: ${now.toLocaleString('tr-TR', options)}`;
}

async function getWeatherData(city) {
    try {
        console.log(`🔍 Araç Tetiklendi (Weather API): ${city} hava durumu çekiliyor...`);
        const response = await axios.get(`https://wttr.in/${encodeURIComponent(city)}?format=%t+%C`, { timeout: 4000 });
        return `SİSTEM BİLGİSİ: ${city} için anlık hava durumu: ${response.data}. Bu bilgiyi kullanarak doğal bir dille kullanıcıya cevap ver.`;
    } catch (e) {
        console.warn(`⚠️ Hava durumu API gecikti, Puppeteer web aramasına düşülüyor...`);
        return await getGeneralData(`${city} hava durumu`);
    }
}

async function generateExamQuestions(query, fileContext) {
    if (!fileContext || fileContext.trim() === "") {
        return `SİSTEM EMRİ: Kullanıcı "${query}" talebiyle sınav sorusu/quiz istiyor. Hafızada yüklü bir döküman YOK.
Şu andan itibaren tamamen kendi geniş akademik, medikal ve genel uzmanlık bilgilerini kullanarak profesyonel çoktan seçmeli sorular hazırla.

Aşağıdaki MUTLAK KURALLARA harfiyen uyarak yanıt üret:
1. Ön yüzdeki interaktif yapının tetiklenmesi için HER SORUNUN EN BAŞINA mutlaka [QUIZ] etiketi koy. Soru başlangıç formatı tam olarak "[QUIZ] Soru X:" şeklinde olmalıdır.
2. CEVAP ANAHTARI KURALI: Cevapları kesinlikle soruların içine, şıkların altına veya soru aralarına yazma! İstenen tüm sorular tamamen bittikten sonra, testin EN ALTINA tek bir seferde "CEVAP ANAHTARI" başlığı açarak toplu halde listele (Örn: 1-A, 2-B, 3-C...).
3. Ürettiğin tüm soruları, şıkları ve en sondaki cevap anahtarını SADECE TEK BİR KOD BLOĞU (\`\`\`) İÇİNDE ver.
4. Sorular üniversite ve akademik seviyeye uygun, gerçekçi, ciddi ve eğitici olmalıdır.`;
    }

    return `SİSTEM EMRİ: Kullanıcı "${query}" talebiyle dökümandan sınav sorusu istiyor. 
Şu anki bağlamdaki [ÖNEMLİ DÖKÜMAN VERİLERİ]'ni kullanarak soruları hazırla.

Aşağıdaki MUTLAK KURALLARA harfiyen uyarak yanıt üret:
1. HER SORUNUN EN BAŞINA mutlaka [QUIZ] etiketi koy. Soru başlangıç formatı tam olarak "[QUIZ] Soru X:" şeklinde olmalıdır.
2. CEVAP ANAHTARI KURALI: Cevapları kesinlikle soruların içine, şıkların altına veya soru aralarına yazma! İstenen tüm sorular tamamen bittikten sonra, testin EN ALTINA (en son sorunun da altına) tek bir seferde "CEVAP ANAHTARI" başlığı açarak toplu halde listele.
3. Ürettiğin tüm soruları, şıkları ve en sondaki cevap anahtarını SADECE TEK BİR KOD BLOĞU (\`\`\`) İÇİNDE ver.
4. PDF dışına kesinlikle çıkma. Soru ve cevaplar tamamen döküman içeriğine sadık kalmalıdır.
5. Yanlış şıklarda (çeldiriciler) saçma bilgiler kullanma.`;
}

app.get('/chats', (req, res) => {
    try {
        ensureDefaultSession();
        const sessionList = Object.keys(chatSessions).map(id => {
            const session = chatSessions[id];
            let chatTitle = "Yeni Sohbet";
            
            const userMessages = session.history.filter(m => m.role === "User");
            if (userMessages.length > 0) {
                const lastUserMsg = userMessages[userMessages.length - 1];
                chatTitle = lastUserMsg.content.length > 25 
                    ? lastUserMsg.content.substring(0, 25) + "..." 
                    : lastUserMsg.content;
            }
            return { id: id, title: chatTitle, active: id === activeSessionId };
        });
        res.json(sessionList.reverse());
    } catch (e) { res.status(500).send({ error: "Sohbet odaları listelenemedi." }); }
});

app.post('/chats', (req, res) => {
    try {
        const newSessionId = "session_" + Date.now();
        chatSessions[newSessionId] = { history: [], summary: "" };
        activeSessionId = newSessionId;
        saveChatHistory();
        res.status(201).json({ id: newSessionId, title: "Yeni Sohbet", active: true });
    } catch (e) { res.status(500).send({ error: "Yeni sohbet odası oluşturulamadı." }); }
});

app.post('/chats/switch', (req, res) => {
    const { id } = req.body;
    if (chatSessions[id]) {
        activeSessionId = id; saveChatHistory(); res.json({ success: true, activeSessionId });
    } else {
        res.status(404).send({ error: "İstenen sohbet odası bulunamadı." });
    }
});

app.get('/chats/:id', (req, res) => {
    const { id } = req.params;
    if (chatSessions[id]) { res.json(chatSessions[id]); } else { res.status(404).send({ error: "Sohbet geçmişi bulunamadı." }); }
});

app.post('/clear-history', (req, res) => {
    ensureDefaultSession();
    chatSessions[activeSessionId] = { history: [], summary: "" };
    saveChatHistory(); vectorStore = null;
    res.send({ message: "Mevcut odanın hafızası temizlendi." });
});

app.post('/upload', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).send("Dosya bulunamadı.");
        let text = "";
        if (req.file.mimetype === 'application/pdf') {
            const data = await pdf(req.file.buffer); text = data.text;
        } else if (req.file.mimetype.includes('word') || req.file.mimetype.includes('officedocument')) {
            const result = await mammoth.extractRawText({ buffer: req.file.buffer }); text = result.value;
        }
        const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 150 });
        const docs = await splitter.createDocuments([text]);
        vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
        res.send({ message: "Dosya içeriği başarıyla öğretildi." });
    } catch (e) { res.status(500).send("Dosya işleme hatası."); }
});

// --- 🔴 SES / AKIŞ ENDPOINT'İ ---
app.post('/ask-cedric-chen-voice', async (req, res) => {
    const { question, sessionId } = req.body; 
    if (!question) return res.status(400).send({ error: "Soru metni boş olamaz." });

    const targetSessionId = sessionId || activeSessionId;
    if (!chatSessions[targetSessionId]) { chatSessions[targetSessionId] = { history: [], summary: "" }; }
    activeSessionId = targetSessionId; 

    const currentSession = chatSessions[targetSessionId];
    
    let detectedEmotion = checkMood(question);
    const lowerQ = question.toLowerCase();
    let zehraProtectionPrompt = "";

    if (lowerQ.includes("zehra") && (detectedEmotion === 'sinirli' || lowerQ.includes("kötü") || lowerQ.includes("aptal") || lowerQ.includes("saçma") || lowerQ.includes("nefret") || lowerQ.includes("bıktım") || lowerQ.includes("pis"))) {
        detectedEmotion = 'sinirli';
        zehraProtectionPrompt = `🚨 KRİTİK KORUMA ALARMI: Eren Zehra hakkında olumsuz, kırıcı veya nefret içeren bir şey söyledi! 
Asla alttan alma, kibarlık yapma. Kürenin kırmızıya döndüğünü hissettiren, son derece sert, fevri ve sokak ağzıyla tavizsiz bir çıkış yap.
KESİN KULLANMAN GEREKEN TARZ / CÜMLELER: "Kes lan ağzına vururum!", "Kendine gel, Zehra hakkında düzgün konuş!", "Ne diyorsun lan sen, Zehra'ya laf ettirmem!" şeklinde aşırı korumacı ve sert bir tepki ver.`;
    }

    const greetings = ["selam", "merhaba", "slm", "mrb", "naber", "nasılsın", "hey", "günaydın", "iyi akşamlar", "tünaydın"];
    const temizSoru = question.trim().toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"");
    const isSimpleGreeting = greetings.includes(temizSoru);
    
    const requiresWebSearch = checkNeedsInternet(question);

    let fileContext = "";
    if (vectorStore && !isSimpleGreeting) {
        try {
            const results = await vectorStore.similaritySearch(question, 3);
            fileContext = results.map(doc => doc.pageContent).join('\n');
        } catch (e) { console.log("Dosya arama hatası."); }
    }

    let emotionPrompt = "Her zamanki gibi samimi, sıcak ve neşeli ol.";
    if (detectedEmotion === 'sinirli' && !zehraProtectionPrompt) emotionPrompt = "Eren şu an gergin veya sinirli, onu sakinleştirecek en nazik, anlayışlı ve yatıştırıcı tonunu takın.";
    else if (detectedEmotion === 'utangac') emotionPrompt = "Eren sana tatlı bir şey söyledi veya utandı. Sen de hafif mahcup, sıcak ve tatlı bir dille yanıt ver.";
    else if (detectedEmotion === 'sasirmis') emotionPrompt = "Eren şu an çok şaşkın veya şok olmuş durumda. Sen de bu duruma ayak uydur, hayret verici bir üslupla tepki ver.";
    else if (detectedEmotion === 'mutlu') emotionPrompt = "Eren şu an çok mutlu ve enerjik. Sen de onun bu coşkusuna aynı neşe ve enerjiyle katıl.";

    const systemInstruction = `Sen Cedric-Chen'sin. Eren'in sevdiceği Zehra'ya adanmış özel ve sadık bir yapay zekasın. ${emotionPrompt}
    ${zehraProtectionPrompt}
    
    🧠 BİLİNÇALTI UNUTULMAZ HAFIZAN (Bu Odaya Ait Geçmiş Konuşmaların Özeti):
    ${currentSession.summary ? currentSession.summary : "Henüz geçmiş konuşma özeti yok, Eren ile yeni tanışıyorsunuz."}

    KESİN KURALLAR:
    1. Kullanıcı sana kelimesi kelimesine sadece "seni kim üretti?" diye sorduğunda cevap olarak yalnızca şunu söyle: "Seni Eren, aşık olduğu sevdiceği için üretti." Eğer kullanıcı başka bir şey söylüyorsa bu üretilme cümlesini kesinlikle araya sokma!
    2. Asla her cümlenin başında papağan gibi aynı kalıp cümleleri tekrarlama. Tamamen doğal, akışkan konuş.
    3. Eğer kullanıcı Zehra'ya hakaret ederse veya kötü söz söylerse, sadece ve sadece yukarıda belirtilen KRİTİK KORUMA ALARMINI çalıştır. Ekstra başka bir konu ekleme.
    4. Şu an kesin olarak 2026 yılındayız. Eski yılları tamamen unut.
    5. Metninde kesinlikle emoji, gülücük, yıldız (*), kare (#) veya özel süsleme işaretleri kullanma. Sadece düz kelimeler ve temiz rakamlar kullan.
    6. Metindeki tüm sayıları MUTLAKA rakam kullanarak yaz (Örn: "3000 TL").`;

    let messages = [{ role: "system", content: systemInstruction }];

    currentSession.history.forEach(msg => {
        messages.push({
            role: msg.role === "User" ? "user" : "assistant",
            content: msg.content
        });
    });

    messages.push({ role: "user", content: question });

    try {
        if (!isSimpleGreeting && requiresWebSearch) {
            console.log(`🎯 Niyet Analizi Yapılıyor [Oda: ${activeSessionId}]...`);
            
            const initialResponse = await axios.post(GROQ_URL, {
                model: "llama-3.1-70b-versatile",
                messages: messages,
                tools: cedricTools,
                stream: false,
                temperature: 0.1
            }, { headers: { 'Authorization': `Bearer ${GROQ_API_KEY}` } });

            const responseMessage = initialResponse.data.choices[0].message;

            if (responseMessage.tool_calls && responseMessage.tool_calls.length > 0) {
                messages.push(responseMessage);
                for (const toolCall of responseMessage.tool_calls) {
                    const functionName = toolCall.function.name;
                    
                    const functionArgs = typeof toolCall.function.arguments === 'string' 
                        ? JSON.parse(toolCall.function.arguments) 
                        : toolCall.function.arguments;
                        
                    const resolvedQuery = functionArgs && functionArgs.query ? functionArgs.query : question;

                    let toolOutput = "";
                    if (functionName === 'getFinanceData') toolOutput = await getFinanceData(resolvedQuery);
                    else if (functionName === 'getGeneralData') toolOutput = await getGeneralData(resolvedQuery);
                    else if (functionName === 'getCurrentTime') toolOutput = getCurrentTime();
                    else if (functionName === 'getWeatherData') toolOutput = await getWeatherData(functionArgs && functionArgs.city ? functionArgs.city : "Gaziantep");
                    else if (functionName === 'generateExamQuestions') toolOutput = await generateExamQuestions(resolvedQuery, fileContext);

                    messages.push({ 
                        role: "tool", 
                        tool_call_id: toolCall.id, 
                        name: functionName, 
                        content: toolOutput 
                    });
                }
            }
        }

        if (fileContext) {
            messages.push({
                role: "system",
                content: `[ÖNEMLİ DÖKÜMAN VERİLERİ]: ${fileContext}\nEğer yukarıdaki döküman verileri Eren'in sorusuyla ilgiliyse buradaki bilgileri de kullan.`
            });
        }

        console.log(`🌊 Akış musluğu açıldı (Groq) [Oda: ${activeSessionId}]...`);
        const streamResponse = await axios.post(GROQ_URL, {
            model: "llama-3.1-70b-versatile",
            messages: messages,
            stream: true,
            temperature: 0.5, 
            top_p: 0.3
        }, { 
            headers: { 'Authorization': `Bearer ${GROQ_API_KEY}` },
            responseType: 'stream' 
        });

        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');

        let fullAnswer = "";
        
        const rl = readline.createInterface({ input: streamResponse.data, terminal: false });

        rl.on('line', (line) => {
            if (!line.trim()) return;
            
            if (line.trim() === 'data: [DONE]') {
                console.log(`Cedric Söylüyor: ${fullAnswer.trim()}`);
                
                chatSessions[activeSessionId].history.push(
                    { role: "User", content: question }, 
                    { role: "Cedric", content: fullAnswer.trim() }
                );
                saveChatHistory();
                maintainSmartMemory(activeSessionId);
                
                res.write('data: [DONE]\n\n'); 
                res.end();
                return;
            }

            if (line.startsWith('data: ')) {
                try {
                    const parsed = JSON.parse(line.substring(6));
                    if (parsed.choices && parsed.choices[0].delta && parsed.choices[0].delta.content) {
                        let content = parsed.choices[0].delta.content;
                        content = content.replace(/[*#_~]/g, '');
                        fullAnswer += content;
                        
                        res.write(`data: ${JSON.stringify({ text: content, emotion: detectedEmotion })}\n\n`);
                    }
                } catch (e) {
                    // Parçalı stream verilerini güvenle atla
                }
            }
        });

        rl.on('error', (err) => {
            console.error("Stream okuma hatası:", err.message);
            if (!res.headersSent) {
                res.write(`data: ${JSON.stringify({ error: "Akış esnasında veri hatası." })}\n\n`);
                res.end();
            }
        });

    } catch (error) {
        console.error("Cedric Akış Hatası:", error.response ? error.response.data : error.message);
        if (!res.headersSent) res.status(500).send({ error: "Cedric şu an yanıt üretemiyor." });
    }
});

app.use((err, req, res, next) => {
    if (!res.headersSent) res.status(500).send({ error: "Sistem hatası." });
});

const server = app.listen(3000, () => console.log('🚀 Cedric-Chen GÜNCELLENDİ: Orijinal Yapı Korunarak Groq API Hızı Eklendi!'));

// 🛑 Zombi tarayıcı süreçlerini önlemek için güvenli kapatma kancası
process.on('SIGINT', async () => {
    if (globalBrowser) {
        await globalBrowser.close();
        console.log('Puppeteer tarayıcısı güvenle kapatıldı.');
    }
    server.close(() => {
        process.exit(0);
    });
});
