require('dotenv').config();
const express = require('express');
const { InferenceClient } = require('@huggingface/inference');
const scheduleData = require('./schedule.json');

const app = express();
app.use(express.json({ limit: '32kb' }));
app.use(express.static('public'));

const PORT = process.env.PORT || 3000;
const HF_TOKEN = process.env.HF_TOKEN;
const MODEL = process.env.HF_MODEL || 'Qwen/Qwen2.5-32B-Instruct';
const PROVIDER = process.env.HF_PROVIDER || 'auto';
const MAX_TOKENS = Number(process.env.HF_MAX_TOKENS || 450);
const RATE_LIMIT = Number(process.env.CHAT_RATE_LIMIT || 12);
const RATE_WINDOW_MS = Number(process.env.CHAT_RATE_WINDOW_MS || 60_000);

const hf = HF_TOKEN ? new InferenceClient(HF_TOKEN) : null;
const rateMap = new Map();
const cache = new Map();

function getClientKey(req) {
    return req.ip || req.headers['x-forwarded-for']?.split(',')[0]?.trim() || 'unknown';
}

function isRateLimited(key) {
    const now = Date.now();
    const recent = (rateMap.get(key) || []).filter(t => now - t < RATE_WINDOW_MS);
    if (recent.length >= RATE_LIMIT) {
        rateMap.set(key, recent);
        return true;
    }
    recent.push(now);
    rateMap.set(key, recent);
    return false;
}

function normalize(text) {
    return String(text || '').trim().toLowerCase().replace(/\s+/g, ' ');
}

function formatClass(c) {
    return `- ${c.time || '-'} | ${c.subject || c.code || '-'} | ห้อง ${c.room || '-'} | กลุ่ม ${c.group || '-'}`;
}

const dayAliases = {
    monday: ['จันทร์', 'วันจันทร์', 'monday'],
    tuesday: ['อังคาร', 'วันอังคาร', 'tuesday'],
    wednesday: ['พุธ', 'วันพุธ', 'wednesday'],
    thursday: ['พฤหัส', 'พฤหัสบดี', 'วันพฤหัส', 'วันพฤหัสบดี', 'thursday'],
    friday: ['ศุกร์', 'วันศุกร์', 'friday']
};

function findDay(message) {
    for (const [day, aliases] of Object.entries(dayAliases)) {
        if (aliases.some(a => message.includes(a))) return day;
    }
    return null;
}

function localScheduleAnswer(message) {
    const q = normalize(message);
    const teacher = scheduleData.teacher_info;
    const schedule = scheduleData.schedule || {};

    if (/^(ขอ)?(ข้อมูล)?อาจารย์|ประวัติอาจารย์|อาจารย์จักรกฤษณ์/.test(q)) {
        return [
            `- ชื่อ: ${teacher.name}`,
            `- วุฒิการศึกษา: ${teacher.degree}`,
            `- ตำแหน่ง: ${teacher.position}`,
            `- สังกัด: ${teacher.college}`
        ].join('\n');
    }

    const day = findDay(q);
    if (day && /(ตาราง|เรียน|สอน|คาบ|วิชา|ห้อง|วันนี้)/.test(q)) {
        const classes = schedule[day] || [];
        if (!classes.length) return '- ไม่มีการเรียนการสอนในวันนี้';
        return [`- ตารางเรียนวัน${day === 'monday' ? 'จันทร์' : day === 'tuesday' ? 'อังคาร' : day === 'wednesday' ? 'พุธ' : day === 'thursday' ? 'พฤหัสบดี' : 'ศุกร์'}`, ...classes.map(formatClass)].join('\n');
    }

    if (/(ตารางเรียนทั้งหมด|ตารางสอนทั้งหมด|ดูตารางทั้งหมด)/.test(q)) {
        const labels = { monday: 'จันทร์', tuesday: 'อังคาร', wednesday: 'พุธ', thursday: 'พฤหัสบดี', friday: 'ศุกร์' };
        return Object.entries(schedule).map(([d, classes]) =>
            `- วัน${labels[d]}: ${classes.length ? classes.map(c => `${c.time} ${c.subject}`).join('; ') : 'ไม่มีการเรียน'}`
        ).join('\n');
    }

    return null;
}

// schedule.json อยู่ที่ root จึงต้องมี route ให้หน้าเว็บเรียกได้
app.get('/schedule.json', (req, res) => {
    res.json(scheduleData);
});

app.get('/api/health', (req, res) => {
    res.json({
        ok: true,
        huggingfaceConfigured: Boolean(HF_TOKEN),
        model: MODEL,
        provider: PROVIDER
    });
});

app.post('/api/chat', async (req, res) => {
    const userMessage = String(req.body?.message || '').trim();

    if (!userMessage) return res.status(400).json({ error: 'กรุณาพิมพ์ข้อความ' });
    if (userMessage.length > 1000) return res.status(400).json({ error: 'ข้อความยาวเกินไป กรุณาย่อคำถาม' });

    const clientKey = getClientKey(req);
    if (isRateLimited(clientKey)) {
        return res.status(429).json({
            error: 'ส่งคำถามถี่เกินไป กรุณารอสักครู่แล้วลองใหม่'
        });
    }

    // คำถามตารางเรียนที่ตอบจาก JSON ไม่ต้องเสียโควตา AI
    const localAnswer = localScheduleAnswer(userMessage);
    if (localAnswer) {
        res.type('text/plain; charset=utf-8').send(localAnswer);
        return;
    }

    if (!hf) {
        return res.status(503).json({ error: 'ยังไม่ได้ตั้งค่า HF_TOKEN บนเซิร์ฟเวอร์' });
    }

    const cacheKey = normalize(userMessage);
    const cached = cache.get(cacheKey);
    if (cached && Date.now() - cached.time < 10 * 60 * 1000) {
        res.type('text/plain; charset=utf-8').send(cached.answer);
        return;
    }

    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('X-Accel-Buffering', 'no');

    try {
        const stream = hf.chatCompletionStream({
            model: MODEL,
            provider: PROVIDER,
            temperature: 0.1,
            max_tokens: MAX_TOKENS,
            messages: [
                {
                    role: 'system',
                    content: `คุณคือ AI ผู้ช่วยตอบคำถามเกี่ยวกับตารางสอนของอาจารย์จักรกฤษณ์ วงศ์อาษา
ข้อมูลอ้างอิง:
${JSON.stringify(scheduleData)}

กฎ:
1. ตอบภาษาไทย กระชับ และตรงคำถาม
2. ใช้ข้อมูลจาก JSON เท่านั้นเมื่อถามเรื่องอาจารย์หรือตารางสอน
3. ห้ามสร้างข้อมูลที่ไม่มีใน JSON
4. ไม่ต้องแสดงกระบวนการคิด
5. หากไม่มีข้อมูล ให้บอกว่าไม่พบข้อมูลในตาราง
6. ตอบเป็น bullet points เมื่อเป็นรายการ
`
                },
                { role: 'user', content: userMessage }
            ]
        });

        let fullAnswer = '';
        for await (const chunk of stream) {
            const text = chunk.choices?.[0]?.delta?.content || '';
            if (text) {
                fullAnswer += text;
                res.write(text);
            }
        }

        if (!fullAnswer) {
            throw new Error('Hugging Face returned an empty response');
        }

        cache.set(cacheKey, { answer: fullAnswer, time: Date.now() });
        // จำกัด cache ไม่ให้โตไม่สิ้นสุด
        if (cache.size > 200) cache.delete(cache.keys().next().value);

        res.end();
    } catch (error) {
        console.error('Hugging Face error:', error);

        if (res.headersSent) {
            res.write('\n\n- ระบบ AI ขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง');
            res.end();
            return;
        }

        const status = Number(error?.status || error?.statusCode || 500);
        if (status === 401 || status === 403) {
            return res.status(503).json({ error: 'HF_TOKEN ไม่ถูกต้องหรือไม่มีสิทธิ์เรียก Inference Providers' });
        }
        if (status === 429) {
            return res.status(429).json({ error: 'โควตาหรืออัตราการเรียก Hugging Face เต็มชั่วคราว กรุณาลองใหม่ภายหลัง' });
        }

        return res.status(502).json({ error: 'เชื่อมต่อ Hugging Face ไม่สำเร็จ กรุณาลองใหม่อีกครั้ง' });
    }
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`HF model: ${MODEL} | provider: ${PROVIDER}`);
});
