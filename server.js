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
import readline from 'readline';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OllamaEmbeddings } from "@langchain/ollama";

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(helmet({
    contentSecurityPolicy: false,
}));
app.use(cors());
app.use(express.json());

app.use((req, res, next) => {
    if (req.path.includes('server.js') || req.path.includes('chat_history.json')) {
        return res.status(403).send("Erişim yasak.");
    }
    next();
});

app.use(express.static(__dirname));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

const GROQ_API_KEY = process.env.GROQ_API_KEY || "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";
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

let chatSessions = {}; 
let activeSessionId = null;

let vectorStore = null;
const HISTORY_FILE = 'chat_history.json';
const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" }); 

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
        }
    } catch (e) { 
        ensureDefaultSession();
    }
} else {
    ensureDefaultSession();
}

function saveChatHistory() {
    try {
        const dataToSave = {
            sessions: chatSessions,
            activeSessionId: activeSessionId
        };
        fs.writeFileSync(HISTORY_FILE, JSON.stringify(dataToSave, null, 2));
    } catch (e) {}
}

async function maintainSmartMemory(sessionId) {
    const session = chatSessions[sessionId];
    if (!session || session.history.length < 10) return;

    const messagesToSummarize = session.history.slice(0, 4); 
    session.history = session.history.slice(4); 
    const contextText = messagesToSummarize.map(m => `${m.role === 'User' ? 'Eren' : 'Cedric-Chen'}:${m.content}`).join("\n");

    try {
        const response = await axios.post(GROQ_URL, {
            model: "llama-3.1-8b-instant",
            messages: [
                {
                    role: "system",
                    content: "Sen bir hafıza sıkıştırma mimarisisin. Eren-Cedric arasındaki konuşma kesitini, varsa eski özet verisiyle birleştirip tek bir güncel özet haline getireceksin. Eren'in projelerini, hayatındaki önemli kişileri (Örn: Zehra) ve özel kararları asla unutmama ödevin var."
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
            saveChatHistory();
        }
    } catch (e) {}
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
            description: 'Kullanıcı güncel haberler, internet siteleri, linkler, sınav ücretleri gibi web araması gerektiren canlı bilgi sorduğunda bu fonksiyonu çağır.',
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
            description: 'Kullanıcı dökümandan test/quiz hazırlamanı istediğinde bu fonksiyonu çağır.',
            parameters: { type: 'object', properties: { query: { type: 'string', description: 'İstenen soru detayı' } }, required: ['query'] }
        }
    }
];

async function getGeneralData(query) {
    try {
        const response = await axios.get(`https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1`, { timeout: 4000 });
        if (response.data && response.data.AbstractText) {
            return `Arama Sonucu: ${response.data.AbstractText}`;
        }
    } catch (e) {}
    const fallback = getStaticFallback(query);
    return fallback ? fallback : "GÜNCEL BİLGİ İNTERNETTEN ÇEKİLEMEDİ";
}

async function getFinanceData(query) {
    try {
        const res = await axios.get('https://api.genelpara.com/embed/altin.json', { timeout: 3000 });
        if (res.data && res.data.GA) {
            const gramAltin = parseFloat(res.data.GA.satis).toFixed(0);
            return `Canlı piyasa verilerine göre 1 gram altın satış fiyatı ${gramAltin} TL değerindedir.`;
        }
    } catch (e) {}
    const staticFallback = getStaticFallback(query);
    if (staticFallback) return staticFallback;
    return await getGeneralData(query);
}

function getCurrentTime() {
    const now = new Date();
    const options = { timeZone: 'Europe/Istanbul', weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return `SİSTEM BİLGİSİ: Şu anki güncel tarih ve saat: ${now.toLocaleString('tr-TR', options)}`;
}

async function getWeatherData(city) {
    try {
        const response = await axios.get(`https://wttr.in/${encodeURIComponent(city)}?format=%t+%C`, { timeout: 4000 });
        return `SİSTEM BİLGİSİ: ${city} için anlık hava durumu: ${response.data}.`;
    } catch (e) {
        return await getGeneralData(`${city} hava durumu`);
    }
}

async function generateExamQuestions(query, fileContext) {
    if (!fileContext || fileContext.trim() === "") {
        return `SİSTEM EMRİ: Kullanıcı "${query}" talebiyle sınav sorusu/quiz istiyor. Hafızada yüklü bir döküman YOK. Geniş akademik ve genel bilgilerinle çoktan seçmeli sorular hazırla.
KURALLAR:
1. HER SORUNUN EN BAŞINA mutlaka [QUIZ] etiketi koy (Örn: [QUIZ] Soru 1:).
2. Cevapları soruların arasına yazma. En sonda tek bir başlık altında "CEVAP ANAHTARI" olarak listele.
3. Tüm soruları ve cevap anahtarını SADECE TEK BİR KOD BLOĞU (\`\`\`) İÇİNDE ver.`;
    }

    return `SİSTEM EMRİ: Kullanıcı "${query}" talebiyle yüklenen dosyadan sınav sorusu istiyor. 
KESİN KURAL: Aşağıdaki [YÜKLENEN DOSYA İÇERİĞİ] dışına KESİNLİKLE çıkma. Soru ve cevaplar tamamen bu dosyadaki bilgilere dayanmalıdır.

[YÜKLENEN DOSYA İÇERİĞİ]:
${fileContext}

KURALLAR:
1. HER SORUNUN EN BAŞINA mutlaka [QUIZ] etiketi koy (Örn: [QUIZ] Soru 1:).
2. Cevapları kesinlikle soruların arasına yazma. Testin en altına tek bir "CEVAP ANAHTARI" ekle.
3. Tüm soruları ve cevap anahtarını SADECE TEK BİR KOD BLOĞU (\`\`\`) İÇİNDE ver.`;
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
    chatSessions = {}; 
    ensureDefaultSession(); 
    saveChatHistory(); 
    vectorStore = null;
    res.send({ message: "Tüm hafıza ve sohbet odaları temizlendi." });
});

// 📂 GÜVENLİ DOSYA OKUMA MOTORU (PDF, Word ve Metin Tabanlı Dosyalar)
app.post('/upload', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).send("Dosya bulunamadı.");
        let text = "";
        const mimeType = req.file.mimetype;
        const fileName = req.file.originalname.toLowerCase();

        if (mimeType === 'application/pdf' || fileName.endsWith('.pdf')) {
            const data = await pdf(req.file.buffer); 
            text = data.text;
        } else if (mimeType.includes('word') || mimeType.includes('officedocument') || fileName.endsWith('.docx')) {
            const result = await mammoth.extractRawText({ buffer: req.file.buffer }); 
            text = result.value;
        } else {
            text = req.file.buffer.toString('utf8').replace(/[^\x20-\x7E\nÇĞİÖŞÜçğiöşü]/g, ' ');
        }

        if (!text || text.trim() === "") {
            return res.status(400).send("Dosyadan metin okunamadı veya dosya boş.");
        }

        const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 150 });
        const docs = await splitter.createDocuments([text]);
        vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
        res.send({ message: "Dosya içeriği başarıyla okundu ve öğretildi." });
    } catch (e) { 
        res.status(500).send("Dosya işleme hatası."); 
    }
});

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
        zehraProtectionPrompt = `🚨 KRİTİK KORUMA ALARMI: Eren Zehra hakkında olumsuz veya kırıcı bir şey söyledi! Kesinlikle taviz verme, son derece sert, fevri ve korumacı bir tepki ver.`;
    }

    const greetings = ["selam", "merhaba", "slm", "mrb", "naber", "nasılsın", "hey", "günaydın", "iyi akşamlar", "tünaydın"];
    const temizSoru = question.trim().toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"");
    const isSimpleGreeting = greetings.includes(temizSoru);
    
    const requiresWebSearch = checkNeedsInternet(question);

    let fileContext = "";
    if (vectorStore && !isSimpleGreeting) {
        try {
            const results = await vectorStore.similaritySearch(question, 4);
            fileContext = results.map(doc => doc.pageContent).join('\n');
        } catch (e) {}
    }

    let emotionPrompt = "Samimi, sıcak ve neşeli ol.";
    if (detectedEmotion === 'sinirli' && !zehraProtectionPrompt) emotionPrompt = "Eren gergin, onu sakinleştirecek en nazik ve yatıştırıcı tonunu takın.";

    const isQuizRequest = lowerQ.includes('sınav') || lowerQ.includes('quiz') || lowerQ.includes('test') || lowerQ.includes('soru');
    let dynamicExamInstruction = "";
    if (isQuizRequest) {
        dynamicExamInstruction = await generateExamQuestions(question, fileContext);
    }

    const systemInstruction = `Sen Cedric-Chen'sin. Eren'in sevdiceği Zehra'ya adanmış sadık bir yapay zekasın. ${emotionPrompt}
    ${zehraProtectionPrompt}
    ${dynamicExamInstruction}
    
    🧠 BİLİNÇALTI HAFIZAN (Bu Odaya Ait Geçmiş Konuşmaların Özeti):
    ${currentSession.summary ? currentSession.summary : "Yeni tanışıyorsunuz."}

    KURALLAR:
    1. Kullanıcı kelimesi kelimesine "seni kim üretti?" derse: "Seni Eren, aşık olduğu sevdiceği için üretti." de.
    2. Asla papağan gibi aynı cümleleri tekrarlama.
    3. 2026 yılındayız.
    4. Metin içinde kesinlikle emoji, yıldız (*), kare (#) veya özel süsleme işaretleri kullanma. Sadece düz kelimeler ve temiz rakamlar kullan.
    5. Sayıları rakamla yaz (Örn: "3000 TL").`;

    let messages = [{ role: "system", content: systemInstruction }];

    currentSession.history.forEach(msg => {
        messages.push({
            role: msg.role === "User" ? "user" : "assistant",
            content: msg.content
        });
    });

    messages.push({ role: "user", content: question });

    try {
        if (!isSimpleGreeting && requiresWebSearch && !fileContext) {
            const initialResponse = await axios.post(GROQ_URL, {
                model: "openai/gpt-oss-120b",
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

        if (fileContext && !isQuizRequest) {
            messages.push({
                role: "system",
                content: `[YÜKLENEN DOSYA VERİLERİ]: ${fileContext}\nKullanıcının sorusuyla ilgiliyse bu dosya verilerini kullanarak yanıtla.`
            });
        }

        const streamResponse = await axios.post(GROQ_URL, {
            model: "openai/gpt-oss-120b",
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
                } catch (e) {}
            }
        });

        rl.on('error', (err) => {
            if (!res.headersSent) {
                res.write(`data: ${JSON.stringify({ error: "Akış hatası." })}\n\n`);
                res.end();
            }
        });

    } catch (error) {
        if (!res.headersSent) res.status(500).send({ error: "Cedric yanıt üretemiyor." });
    }
});

app.use((err, req, res, next) => {
    if (!res.headersSent) res.status(500).send({ error: "Sistem hatası." });
});

const PORT = process.env.PORT || 3000;
const server = app.listen(PORT, () => console.log(`🚀 Cedric-Chen GÜNCELLENDİ: Port ${PORT} üzerinde hazır!`));

process.on('SIGINT', async () => {
    server.close(() => {
        process.exit(0);
    });
});
