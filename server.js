require('dotenv').config();
const express = require('express');
const { InferenceClient } = require('@huggingface/inference');
const scheduleData = require('./schedule.json');

const app = express();
app.use(express.json({ limit: '32kb' }));
app.use(express.static('public'));

const PORT = process.env.PORT || 3000;
const HF_TOKEN = process.env.HF_TOKEN;
const MODEL = process.env.HF_MODEL || 'Qwen/Qwen3-30B-A3B';
const PROVIDER = process.env.HF_PROVIDER || 'auto';

// รายชื่อสำรองสำหรับกรณี model ที่ตั้งไว้ไม่มี provider ให้ token นี้ใช้งาน
const MODEL_CANDIDATES = [
    MODEL,
    'Qwen/Qwen3-30B-A3B',
    'Qwen/Qwen3-32B',
    'openai/gpt-oss-20b'
];
let resolvedModel = MODEL;
let resolvedProvider = PROVIDER;
const MAX_TOKENS = Number(process.env.HF_MAX_TOKENS || 450);
const RATE_LIMIT = Number(process.env.CHAT_RATE_LIMIT || 12);
const RATE_WINDOW_MS = Number(process.env.CHAT_RATE_WINDOW_MS || 60_000);

const hf = HF_TOKEN ? new InferenceClient(HF_TOKEN) : null;
const rateMap = new Map();
const cache = new Map();

async function resolveAvailableModel() {
    if (!HF_TOKEN) return;

    try {
        const response = await fetch('https://router.huggingface.co/v1/models', {
            headers: { Authorization: 'Bearer ' + HF_TOKEN }
        });

        if (!response.ok) {
            console.warn('Could not inspect Hugging Face available models:', response.status);
            return;
        }

        const payload = await response.json();
        const models = Array.isArray(payload?.data) ? payload.data : [];

        for (const candidate of MODEL_CANDIDATES) {
            const entry = models.find(m => m?.id === candidate);
            if (!entry) continue;

            const providers = Array.isArray(entry.providers) ? entry.providers : [];
            if (PROVIDER === 'auto' || PROVIDER === 'default') {
                const live = providers.find(p => p?.status === 'live');
                if (live || providers.length) {
                    resolvedModel = candidate;
                    resolvedProvider = 'auto';
                    console.log('HF model selected:', resolvedModel, '| provider: auto');
                    return;
                }
            } else {
                const match = providers.find(p => p?.provider === PROVIDER && p?.status !== 'disabled');
                if (match) {
                    resolvedModel = candidate;
                    resolvedProvider = PROVIDER;
                    console.log('HF model selected:', resolvedModel, '| provider:', resolvedProvider);
                    return;
                }
            }
        }

        console.warn('No configured Hugging Face model/provider was found. Requested:', MODEL, '| provider:', PROVIDER);
    } catch (error) {
        console.warn('Could not resolve Hugging Face model:', error.message);
    }
}

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
    return String(text || '').trim().toLowerCase()
        .replace(/[!?.,;:()[\]{}\"“”‘’]/g, ' ')
        .replace(/\s+/g, ' ');
}

const DAY_LABELS = { monday: 'จันทร์', tuesday: 'อังคาร', wednesday: 'พุธ', thursday: 'พฤหัสบดี', friday: 'ศุกร์' };
const DAY_KEYS = Object.keys(DAY_LABELS);
const dayAliases = { monday: ['จันทร์', 'วันจันทร์', 'จัน', 'monday'], tuesday: ['อังคาร', 'วันอังคาร', 'tuesday'], wednesday: ['พุธ', 'วันพุธ', 'wednesday'], thursday: ['พฤหัส', 'พฤหัสบดี', 'วันพฤหัส', 'วันพฤหัสบดี', 'thursday'], friday: ['ศุกร์', 'วันศุกร์', 'friday'] };
const englishDayToKey = { Monday: 'monday', Tuesday: 'tuesday', Wednesday: 'wednesday', Thursday: 'thursday', Friday: 'friday' };

function getThailandDateParts() {
    const parts = new Intl.DateTimeFormat('en-US', { timeZone: 'Asia/Bangkok', weekday: 'long', hour: '2-digit', minute: '2-digit', hour12: false }).formatToParts(new Date());
    const out = {};
    for (const p of parts) out[p.type] = p.value;
    return { day: englishDayToKey[out.weekday] || null, minutes: Number(out.hour || 0) * 60 + Number(out.minute || 0) };
}

function shiftWeekday(day, offset) { const index = DAY_KEYS.indexOf(day); if (index < 0) return null; return DAY_KEYS[(index + offset + DAY_KEYS.length) % DAY_KEYS.length]; }

function findDay(message) {
    const q = normalize(message); const today = getThailandDateParts().day;
    if (q.includes('วันนี้')) return today;
    if (q.includes('พรุ่งนี้') || q.includes('พรุ่ง')) return shiftWeekday(today, 1);
    if (q.includes('มะรืน')) return shiftWeekday(today, 2);
    for (const [day, aliases] of Object.entries(dayAliases)) if (aliases.some(a => q.includes(a))) return day;
    return null;
}

function getDayClasses(day) { return [...(scheduleData.schedule?.[day] || [])].sort((a, b) => String(a.time || '').localeCompare(String(b.time || ''), undefined, { numeric: true })); }
function parseTimeRange(time) { const m = String(time || '').match(/(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})/); return m ? { start: Number(m[1]) * 60 + Number(m[2]), end: Number(m[3]) * 60 + Number(m[4]) } : null; }
function minutesToTime(minutes) { return String(Math.floor(minutes / 60)).padStart(2, '0') + ':' + String(minutes % 60).padStart(2, '0'); }
function formatClass(c) { return '- ' + (c.time || '-') + ' | ' + (c.subject || c.code || '-') + ' | ห้อง ' + (c.room || '-') + ' | กลุ่ม ' + (c.group || '-'); }
function formatClasses(classes) { return classes.length ? classes.map(formatClass).join('\n') : '- ไม่มีการเรียนการสอน'; }

function findSubject(q) {
    const subjects = scheduleData.subjects || {};
    for (const [code, subject] of Object.entries(subjects)) if (q.includes(String(code).toLowerCase()) || q.includes(normalize(subject))) return { code, subject };
    for (const classes of Object.values(scheduleData.schedule || {})) for (const c of classes) if (c.code && q.includes(String(c.code).toLowerCase())) return { code: c.code, subject: c.subject };
    return null;
}

function getFreeSlots(classes) {
    const ranges = classes.map(c => parseTimeRange(c.time)).filter(Boolean).sort((a, b) => a.start - b.start);
    if (!ranges.length) return ['08:00-18:00'];
    const slots = []; let cursor = 8 * 60;
    for (const r of ranges) { if (r.start > cursor) slots.push(minutesToTime(cursor) + '-' + minutesToTime(r.start)); cursor = Math.max(cursor, r.end); }
    if (cursor < 18 * 60) slots.push(minutesToTime(cursor) + '-18:00');
    return slots;
}

function localScheduleAnswer(message) {
    const q = normalize(message); const teacher = scheduleData.teacher_info || {}; const schedule = scheduleData.schedule || {};
    const scheduleWords = /(ตาราง|เรียน|สอน|คาบ|วิชา|ห้อง|กลุ่ม|ว่าง|เลิก|เริ่ม|กี่โมง|เมื่อไหร่|กี่คาบ|วันนี้|พรุ่งนี้|มะรืน|เช้า|บ่าย|เย็น)/;

    if (/(ข้อมูล|ประวัติ|โปรไฟล์|เกี่ยวกับ).{0,12}(อาจารย์|ครู|ผู้สอน)/.test(q) || /(อาจารย์|ครู|ผู้สอน).{0,12}(ชื่อ|จบ|วุฒิ|ตำแหน่ง|สังกัด|อยู่ที่ไหน)/.test(q) || /^(ขอ)?(ข้อมูล)?อาจารย์/.test(q)) {
        return ['- ชื่อ: ' + (teacher.name || '-'), '- วุฒิการศึกษา: ' + (teacher.degree || '-'), '- ตำแหน่ง: ' + (teacher.position || '-'), '- สังกัด: ' + (teacher.college || '-')].join('\n');
    }
    if (/(ตารางเรียนทั้งหมด|ตารางสอนทั้งหมด|ตารางทั้งหมด|ดูตาราง|ตารางอาจารย์)/.test(q)) return DAY_KEYS.map(day => '- วัน' + DAY_LABELS[day] + ': ' + formatClasses(getDayClasses(day)).replace(/^- /, '')).join('\n');

    const day = findDay(q); const subject = findSubject(q);
    if (subject) {
        const occurrences = Object.entries(schedule).flatMap(([d, classes]) => classes.filter(c => c.code === subject.code || normalize(c.subject) === normalize(subject.subject)).map(c => ({ day: d, ...c })));
        if (day) { const dayOccurrences = occurrences.filter(c => c.day === day); if (!dayOccurrences.length) return '- วัน' + DAY_LABELS[day] + 'ไม่มีวิชา' + subject.subject; return ['- ' + subject.subject + ' (' + subject.code + ') วัน' + DAY_LABELS[day], ...dayOccurrences.map(formatClass)].join('\n'); }
        if (/(วันไหน|เมื่อไหร่|ตอนไหน|กี่โมง|เวลา|สอนวัน|เรียนวัน)/.test(q)) return occurrences.map(c => '- วัน' + DAY_LABELS[c.day] + ' ' + formatClass(c)).join('\n') || '- ไม่พบวิชานี้ในตาราง';
        if (/(ห้อง|เรียนที่ไหน|อยู่ไหน)/.test(q)) return occurrences.map(c => '- วัน' + DAY_LABELS[c.day] + ' ' + (c.time || '-') + ' ห้อง ' + (c.room || '-')).join('\n') || '- ไม่พบวิชานี้ในตาราง';
        return occurrences.map(c => '- วัน' + DAY_LABELS[c.day] + ' ' + formatClass(c)).join('\n') || '- ไม่พบวิชานี้ในตาราง';
    }

    if (day && scheduleWords.test(q)) {
        let classes = getDayClasses(day);
        if (/(เช้า|ตอนเช้า)/.test(q)) classes = classes.filter(c => (parseTimeRange(c.time)?.start ?? 9999) < 12 * 60);
        if (/(บ่าย|ตอนบ่าย)/.test(q)) classes = classes.filter(c => (parseTimeRange(c.time)?.start ?? -1) >= 12 * 60);
        if (/(เย็น|ตอนเย็น)/.test(q)) classes = classes.filter(c => (parseTimeRange(c.time)?.start ?? -1) >= 17 * 60);
        if (!classes.length) return '- วัน' + DAY_LABELS[day] + 'ไม่มีคาบตามช่วงเวลาที่ถามครับ';
        if (/(คาบแรก|แรกสุด|เริ่มกี่โมง|เริ่มเรียน)/.test(q)) return '- คาบแรกวัน' + DAY_LABELS[day] + 'เริ่ม ' + classes[0].time;
        if (/(คาบสุดท้าย|สุดท้าย|เลิกกี่โมง|เลิกเรียน)/.test(q)) return '- คาบสุดท้ายวัน' + DAY_LABELS[day] + 'คือ ' + classes[classes.length - 1].subject + ' (' + classes[classes.length - 1].time + ')';
        if (/(กี่คาบ|กี่วิชา|กี่ครั้ง)/.test(q)) return '- วัน' + DAY_LABELS[day] + 'มี ' + classes.length + ' คาบ';
        if (/(ว่าง|มีช่วงว่าง|พัก|ไม่มีเรียน)/.test(q)) return ['- วัน' + DAY_LABELS[day] + 'ช่วงที่ว่าง', ...getFreeSlots(getDayClasses(day)).map(s => '- ' + s)].join('\n');
        return ['- ตารางวัน' + DAY_LABELS[day], ...classes.map(formatClass)].join('\n');
    }

    if (/(ตอนนี้|ขณะนี้|คาบต่อไป|ต่อไปเรียนอะไร|กำลังสอน)/.test(q)) {
        const now = getThailandDateParts(); const classes = getDayClasses(now.day);
        const current = classes.find(c => { const r = parseTimeRange(c.time); return r && now.minutes >= r.start && now.minutes < r.end; });
        if (current) return '- ตอนนี้กำลังสอน: ' + current.subject + ' เวลา ' + current.time + ' ห้อง ' + (current.room || '-');
        const next = classes.find(c => { const r = parseTimeRange(c.time); return r && r.start > now.minutes; });
        if (next) return '- คาบถัดไป: ' + next.subject + ' เวลา ' + next.time + ' ห้อง ' + (next.room || '-');
        return '- ตอนนี้ไม่มีคาบสอนแล้วสำหรับวัน' + DAY_LABELS[now.day];
    }
    if (/(วันนี้|ตอนนี้).{0,20}(ว่าง|มีเวลา|ไม่มีเรียน)/.test(q) || /(^|\s)ว่างไหม($|\s)/.test(q)) { const today = getThailandDateParts().day; return ['- ช่วงว่างวัน' + DAY_LABELS[today], ...getFreeSlots(getDayClasses(today)).map(s => '- ' + s)].join('\n'); }
    if (/(หลัง|หลังจาก).{0,15}(คาบสุดท้าย|เลิกเรียน|เลิกสอน)/.test(q)) { const targetDay = day || getThailandDateParts().day; const classes = getDayClasses(targetDay); if (!classes.length) return '- วัน' + DAY_LABELS[targetDay] + 'ไม่มีคาบสอน'; const last = classes[classes.length - 1]; return '- คาบสุดท้ายวัน' + DAY_LABELS[targetDay] + 'จบ ' + ((last.time || '').split('-')[1] || '-') + '\n- หลังจากนั้นไม่มีคาบในตารางครับ'; }
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
        model: resolvedModel,
        provider: resolvedProvider
    });
});

app.post('/api/chat', async (req, res) => {
    const userMessage = String(req.body?.message || '').trim();
    const history = Array.isArray(req.body?.history)
        ? req.body.history.filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string').slice(-8)
        : [];

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

    const cacheKey = normalize(userMessage) + '|' + history.map(m => m.role + ':' + normalize(m.content)).join('|');
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
            model: resolvedModel,
            provider: resolvedProvider,
            temperature: 0.1,
            max_tokens: MAX_TOKENS,
            messages: [
                {
                    role: 'system',
                    content: `คุณคือ AI ผู้ช่วยตอบคำถามเกี่ยวกับตารางสอนของอาจารย์จักรกฤษณ์ วงศ์อาษา\nข้อมูลอ้างอิง:\n${JSON.stringify(scheduleData)}\n\nกฎสำคัญ:\n1. ผู้ใช้ชอบถามแบบภาษาพูด คำย่อ คำถามกวนๆ หรือประโยคไม่เป็นทางการ เช่น "พรุ่งนี้มีไร", "ครูว่างปะ", "ลินุกซ์เรียนตอนไหนอะ" ให้ตีความเจตนาจากบริบท\n2. ตอบภาษาไทย กระชับ เป็นธรรมชาติ และตอบสิ่งที่ผู้ใช้ต้องการจริงๆ\n3. เรื่องตารางสอน/ข้อมูลผู้สอน ให้ใช้ข้อมูลใน JSON เท่านั้น ห้ามแต่งวัน เวลา ห้อง วิชา หรือข้อมูลส่วนตัวขึ้นเอง\n4. ถ้าคำถามกำกวมจริงๆ ให้ถามกลับสั้นๆ เพื่อขอวัน/วิชา/ช่วงเวลา แทนการเดา\n5. ห้ามแสดงกระบวนการคิด\n6. ถ้าไม่มีข้อมูลใน JSON ให้บอกตรงๆ ว่าไม่พบข้อมูล\n7. ถ้าเป็นคำถามเล่นๆ ที่ยังเกี่ยวกับตาราง ให้ตอบแบบเป็นกันเองได้ แต่ห้ามเปลี่ยนข้อเท็จจริง\n`
                },
                ...history,
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
        if (status === 400 && String(error?.message || '').includes('not supported by any provider')) {
            return res.status(503).json({
                error: 'โมเดลนี้ไม่มี Inference Provider ที่ token ของเซิร์ฟเวอร์เปิดใช้งานอยู่ กรุณาตรวจสอบ HF_MODEL/HF_PROVIDER หรือเปิด provider ใน Hugging Face'
            });
        }
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
