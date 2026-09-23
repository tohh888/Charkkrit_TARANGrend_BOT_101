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
// Fallback หลายชั้น: ถ้าโมเดลหนึ่งว่าง/ล่ม/ไม่มี provider ให้ลองตัวถัดไป
// รายชื่อเน้นโมเดลที่เหมาะกับงานสนทนาและมีโอกาสมี Inference Provider ให้เลือก
const MODEL_CANDIDATES = [
    MODEL,
    'Qwen/Qwen3-30B-A3B',
    'Qwen/Qwen3-32B',
    'Qwen/Qwen3-14B',
    'Qwen/Qwen3-8B',
    'openai/gpt-oss-20b',
    'openai/gpt-oss-120b',
    'meta-llama/Llama-3.3-70B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3'
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
function formatClass(c) {
    return '- ' + (c.time || '-') + ' | ' + (c.subject || c.code || '-') + ' | ห้อง ' + (c.room || '-') + ' | กลุ่ม ' + (c.group || '-');
}
function formatClassComplete(c) {
    return [
        'เวลา: ' + (c.time || '-'),
        'รหัสวิชา: ' + (c.code || '-'),
        'วิชา: ' + (c.subject || '-'),
        'ประเภท: ' + (c.type || '-'),
        'ห้อง: ' + (c.room || '-'),
        'กลุ่ม: ' + (c.group || '-')
    ].join(' | ');
}
function formatClasses(classes) {
    return classes.length ? classes.map(formatClass).join('\n') : '- ไม่มีการเรียนการสอน';
}

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

function allScheduleItems() {
    return DAY_KEYS.flatMap(day => getDayClasses(day).map(c => ({ day, ...c })));
}

function uniqueBy(values, keyFn = x => x) {
    const seen = new Set();
    return values.filter(v => {
        const k = keyFn(v);
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
    });
}

function findGroup(q) {
    const groups = uniqueBy(allScheduleItems().map(x => x.group).filter(Boolean));
    const nq = normalize(q);
    return groups.sort((a, b) => b.length - a.length)
        .find(g => nq.includes(normalize(g))) || null;
}

function groupItems(group) {
    return allScheduleItems().filter(x => normalize(x.group) === normalize(group));
}

function subjectItems(subject) {
    return allScheduleItems().filter(x =>
        x.code === subject.code || normalize(x.subject) === normalize(subject.subject)
    );
}

function minutesForClasses(classes) {
    return classes.reduce((sum, c) => {
        const r = parseTimeRange(c.time);
        return sum + (r ? Math.max(0, r.end - r.start) : 0);
    }, 0);
}

function formatDuration(minutes) {
    const h = Math.floor(minutes / 60);
    const m = minutes % 60;
    return (h ? h + ' ชั่วโมง' : '') + (h && m ? ' ' : '') + (m ? m + ' นาที' : '') || '0 นาที';
}

function formatDaySummary(stats) {
    return stats.map(x => '• ' + x.label + ' — ' + x.count + ' คาบ / ' + formatDuration(x.minutes)).join('\n');
}

function localScheduleAnswer(message, history = []) {
    const q = normalize(message); const teacher = scheduleData.teacher_info || {}; const schedule = scheduleData.schedule || {};
    const scheduleWords = /(ตาราง|เรียน|สอน|คาบ|วิชา|ห้อง|กลุ่ม|ว่าง|เลิก|เริ่ม|กี่โมง|เมื่อไหร่|กี่คาบ|วันนี้|พรุ่งนี้|มะรืน|วันจันทร์|วันอังคาร|วันพุธ|วันพฤหัส|วันพฤหัสบดี|วันศุกร์|จันทร์|อังคาร|พุธ|พฤหัส|ศุกร์|เช้า|บ่าย|เย็น)/;

    // จำบริบทจากคำถามก่อนหน้า เพื่อรองรับคำถามต่อเนื่อง
    const recentUserMessages = history
        .filter(m => m?.role === 'user' && typeof m.content === 'string')
        .map(m => m.content)
        .slice(-4);

    let contextDay = null;
    let contextSubject = null;
    for (let i = recentUserMessages.length - 1; i >= 0; i--) {
        if (!contextDay) contextDay = findDay(recentUserMessages[i]);
        if (!contextSubject) contextSubject = findSubject(normalize(recentUserMessages[i]));
        if (contextDay && contextSubject) break;
    }

    const explicitDay = findDay(q);
    const day = explicitDay || (
        /(วันนั้น|วันดังกล่าว|วันเดิม|แล้ววันนั้น|วันนั้นล่ะ)/.test(q)
            ? contextDay
            : null
    );

    const subject = findSubject(q) || (
        /(วิชานั้น|วิชาเดิม|อันนั้น|ตัวนั้น|วิชานี้)/.test(q)
            ? contextSubject
            : null
    );

    if (/(ข้อมูล|ประวัติ|โปรไฟล์|เกี่ยวกับ).{0,12}(อาจารย์|ครู|ผู้สอน)/.test(q) || /(อาจารย์|ครู|ผู้สอน).{0,12}(ชื่อ|จบ|วุฒิ|ตำแหน่ง|สังกัด|อยู่ที่ไหน)/.test(q) || /^(ขอ)?(ข้อมูล)?อาจารย์/.test(q)) {
        return [
            'ข้อมูลผู้สอน',
            '- ชื่อ: ' + (teacher.name || '-'),
            '- วุฒิการศึกษา: ' + (teacher.degree || '-'),
            '- แผนก/สาขา: ' + (teacher.department || '-'),
            '- ตำแหน่ง: ' + (teacher.position || '-'),
            '- สังกัด: ' + (teacher.college || '-'),
            '- ภาคเรียน: ' + (teacher.term || '-'),
            '- ชั่วโมงทฤษฎี: ' + (teacher.total_theory_hours ?? '-') + ' ชั่วโมง',
            '- ชั่วโมงปฏิบัติ: ' + (teacher.total_practical_hours ?? '-') + ' ชั่วโมง',
            '- หน่วยกิต: ' + (teacher.total_credits ?? '-') ,
            '- ชั่วโมงรวม: ' + (teacher.total_hours ?? '-') + ' ชั่วโมง'
        ].join('\n');
    }
    // เครื่องวิเคราะห์ตารางแบบ deterministic: คำถามที่คำนวณได้ต้องคำนวณจากข้อมูลจริง
    const dayStats = DAY_KEYS.map(d => {
        const classes = getDayClasses(d);
        const minutes = minutesForClasses(classes);
        const freeMinutes = Math.max(0, 10 * 60 - minutes);
        return { day: d, label: 'วัน' + DAY_LABELS[d], classes, count: classes.length, minutes, freeMinutes };
    });

    const hasCountQuestion = /(กี่คาบ|จำนวนคาบ|นับคาบ|คาบทั้งหมด|สอนกี่ครั้ง|กี่ครั้งที่สอน|มีกี่คาบ|กี่รายการ)/.test(q);
    const asksLeast = /(น้อยที่สุด|น้อยสุด|น้อยกว่า|เบาสุด|สอนน้อย|เรียนน้อย|คาบน้อย|เวลาสอนน้อย)/.test(q);
    const asksMost = /(มากที่สุด|มากสุด|มากกว่า|เยอะที่สุด|เยอะสุด|หนักสุด|สอนเยอะ|เรียนเยอะ|คาบเยอะ|เวลาสอนมาก)/.test(q);
    const asksComparison = /(เทียบ|เปรียบเทียบ|ต่างกัน|ห่างกัน|เรียงจาก|เรียงลำดับ|อันดับ|มากกว่ากัน|น้อยกว่ากัน)/.test(q);
    const asksHours = /(ชั่วโมง|ชม\.|เวลาเรียนรวม|เวลาสอนรวม|ใช้เวลาสอน|สอนนาน)/.test(q);
    const metric = asksHours ? 'minutes' : 'count';

    if ((asksLeast || asksMost) && /(วัน|วันไหน|แต่ละวัน|ทุกวัน|วันทำงาน)/.test(q)) {
        const target = asksLeast
            ? Math.min(...dayStats.map(x => x[metric]))
            : Math.max(...dayStats.map(x => x[metric]));
        const matches = dayStats.filter(x => x[metric] === target);
        const label = asksLeast ? 'น้อยที่สุด' : 'มากที่สุด';
        const lines = ['📊 สรุปจากตารางจริง', '', 'วันสอน' + label + ':'];

        lines.push(...matches.map(x =>
            '• ' + x.label + ' — ' +
            (metric === 'minutes' ? formatDuration(x.minutes) : x.count + ' คาบ') +
            ' (' + (metric === 'minutes' ? x.count + ' คาบ' : formatDuration(x.minutes)) + ')'
        ));

        lines.push('', 'ตรวจครบทุกวัน:', formatDaySummary(dayStats));
        return lines.join('\n');
    }

    if (asksComparison && /(คาบ|สอน|ตาราง|ชั่วโมง|เวลา)/.test(q)) {
        const ordered = [...dayStats].sort((a, b) => b[metric] - a[metric]);
        return [
            '📊 เปรียบเทียบตารางสอน',
            '',
            ...ordered.map((x, i) =>
                (i + 1) + '. ' + x.label + ' — ' +
                (metric === 'minutes' ? formatDuration(x.minutes) : x.count + ' คาบ') +
                ' (' + formatDuration(x.minutes) + ')'
            )
        ].join('\n');
    }

    if (hasCountQuestion && !day && !asksLeast && !asksMost) {
        const totalCount = dayStats.reduce((sum, x) => sum + x.count, 0);
        const totalMinutes = dayStats.reduce((sum, x) => sum + x.minutes, 0);
        return [
            '📊 สรุปจำนวนคาบสอน',
            '',
            'รวมทั้งหมด ' + totalCount + ' คาบ',
            'เวลาสอนรวม ' + formatDuration(totalMinutes),
            '',
            formatDaySummary(dayStats)
        ].join('\n');
    }

    if (/(วันไหน|แต่ละวัน|ทุกวัน).*(ว่างที่สุด|ว่างสุด|มีเวลาว่างมากที่สุด|มีเวลาว่างเยอะที่สุด|ว่างมากสุด)/.test(q)) {
        const target = Math.max(...dayStats.map(x => x.freeMinutes));
        const matches = dayStats.filter(x => x.freeMinutes === target);
        return [
            '🕐 วันที่มีเวลาว่างมากที่สุด',
            '',
            ...matches.map(x => '• ' + x.label + ' — ว่างประมาณ ' + formatDuration(x.freeMinutes)),
            '',
            'สรุป: ' + formatDaySummary(dayStats)
        ].join('\n');
    }

    if (day && /(ชั่วโมง|ชม\.|เวลาสอนรวม|สอนกี่ชั่วโมง|ใช้เวลาสอน)/.test(q)) {
        const s = dayStats.find(x => x.day === day);
        return [
            '⏱️ วัน' + DAY_LABELS[day],
            '',
            'สอน ' + s.count + ' คาบ',
            'เวลาสอนรวม ' + formatDuration(s.minutes),
            '',
            ...s.classes.map(c => '• ' + c.time + ' — ' + c.subject)
        ].join('\n');
    }

    if (/(ตารางเรียนทั้งหมด|ตารางสอนทั้งหมด|ตารางทั้งหมด|ดูตาราง|ตารางอาจารย์)/.test(q)) {
        return DAY_KEYS.map(d => {
            const classes = getDayClasses(d);
            return [
                'วัน' + DAY_LABELS[d] + ' (' + classes.length + ' คาบ)',
                ...(classes.length ? classes.map(formatClassComplete) : ['ไม่มีการเรียนการสอน'])
            ].join('\n');
        }).join('\n\n');
    }

    // คำถามเรื่องกลุ่มต้องตอบจากตารางจริงโดยตรง และคืนกลุ่มไม่ซ้ำทั้งหมด
    if (/(กลุ่ม|นักเรียน|นักศึกษา|ผู้เรียน|ห้องเรียน)/.test(q)) {
        const allClasses = DAY_KEYS.flatMap(d =>
            getDayClasses(d).map(c => ({ day: d, ...c }))
        );

        const requestedGroup = allClasses
            .map(c => c.group)
            .filter(Boolean)
            .sort((a, b) => b.length - a.length)
            .find(g => q.includes(normalize(g)));

        // วิเคราะห์ความหมายเชิงลึกของคำถามภาษาไทย:
        // "กลุ่มไหนเรียนอะไร", "แต่ละกลุ่มเรียนวิชาอะไร", "มีกลุ่มไหนเรียนอะไรบ้าง"
        // ต้องการทั้ง "ชื่อกลุ่ม + รายวิชาที่เรียน" ไม่ใช่แค่รายชื่อกลุ่ม
        const asksGroupAndSubjects = (
            /(กลุ่มไหน|กลุ่มอะไร|มีกลุ่ม|แต่ละกลุ่ม|ทุกกลุ่ม|กลุ่มบ้าง|นักเรียน.*กลุ่ม|นักศึกษา.*กลุ่ม|ผู้เรียน.*กลุ่ม)/.test(q)
            && /(เรียนอะไร|เรียนวิชาอะไร|เรียนวิชาไหน|มีวิชาอะไร|มีวิชาไหน|สอนอะไร|สอนวิชาอะไร|วิชาอะไรบ้าง|เรียนบ้าง|มีอะไรบ้าง)/.test(q)
        );

        if (!requestedGroup && asksGroupAndSubjects) {
            const grouped = new Map();

            for (const d of DAY_KEYS) {
                for (const item of getDayClasses(d)) {
                    const group = String(item.group || '').trim();
                    if (!group) continue;

                    if (!grouped.has(group)) grouped.set(group, []);
                    grouped.get(group).push({
                        day: d,
                        ...item
                    });
                }
            }

            if (!grouped.size) return 'ไม่พบข้อมูลกลุ่มและรายวิชาในตาราง';

            const lines = ['📚 ข้อมูลกลุ่มและรายวิชาที่เรียนทั้งหมด'];

            for (const [group, classes] of grouped) {
                lines.push('');
                lines.push('👥 กลุ่ม ' + group);
                lines.push('────────────────────');

                // แยกแต่ละคาบเป็นบล็อก เพื่อให้อ่านง่ายและไม่ติดกันเป็นย่อหน้ายาว
                for (const item of classes) {
                    lines.push(
                        '📅 วัน' + DAY_LABELS[item.day],
                        '🕐 เวลา: ' + (item.time || '-'),
                        '📘 วิชา: ' + (item.subject || '-'),
                        '🔢 รหัสวิชา: ' + (item.code || '-'),
                        '📌 ประเภท: ' + (item.type || '-'),
                        '🏫 ห้อง: ' + (item.room || '-'),
                        '👥 กลุ่ม: ' + (item.group || '-'),
                        ''
                    );
                }
            }

            return lines.join('\n').replace(/\n{3,}/g, '\n\n').trim();
        }

        const asksGroupList = /(กลุ่มไหน|กลุ่มอะไร|มีกลุ่ม|กลุ่มบ้าง|สอนกลุ่ม|สอน.*กลุ่ม|นักเรียน.*กลุ่ม|นักศึกษา.*กลุ่ม|ผู้เรียน.*กลุ่ม)/.test(q);

        if (!requestedGroup && asksGroupList) {
            const groups = [...new Set(allClasses.map(c => c.group).filter(Boolean))];
            return [
                'กลุ่มนักเรียน/นักศึกษาที่สอนทั้งหมด',
                ...groups.map((group, index) => (index + 1) + '. ' + group)
            ].join('\n');
        }

        if (requestedGroup) {
            const matches = groupItems(requestedGroup);
            if (matches.length) {
                const uniqueSubjects = uniqueBy(matches.map(c => ({
                    code: c.code,
                    subject: c.subject
                })), x => x.code || normalize(x.subject));

                if (/(กี่วิชา|จำนวนวิชา|เรียนกี่วิชา|มีกี่วิชา)/.test(q)) {
                    return [
                        '👥 ' + requestedGroup,
                        '',
                        'เรียนทั้งหมด ' + uniqueSubjects.length + ' วิชา',
                        ...uniqueSubjects.map((x, i) => (i + 1) + '. ' + x.subject + ' (' + (x.code || '-') + ')')
                    ].join('\n');
                }

                if (/(เรียนอะไร|เรียนวิชาอะไร|เรียนวิชาไหน|มีวิชาอะไร|วิชาอะไรบ้าง|สอนอะไร|เรียนบ้าง)/.test(q)) {
                    return [
                        '👥 ' + requestedGroup,
                        '',
                        'รายวิชาที่เรียน:',
                        ...uniqueSubjects.map(x => '• ' + x.subject + ' (' + (x.code || '-') + ')'),
                        '',
                        'รายละเอียดคาบ:',
                        ...matches.map(c => '• วัน' + DAY_LABELS[c.day] + ' ' + c.time + ' — ' + c.subject)
                    ].join('\n');
                }

                return [
                    '👥 ข้อมูลกลุ่ม ' + requestedGroup,
                    '',
                    ...matches.map(c => '• วัน' + DAY_LABELS[c.day] + ' | ' + formatClassComplete(c))
                ].join('\n');
            }
        }
    }
    // คำถามเชิงสรุปเกี่ยวกับวิชา/กลุ่ม/ห้อง/ประเภท โดยไม่ต้องพึ่ง AI
    const allItems = allScheduleItems();
    const asksSubjectsList = /(วิชาอะไรบ้าง|มีวิชาอะไร|สอนวิชาอะไร|สอนอะไรบ้าง|เรียนวิชาอะไร|รายวิชาอะไร|วิชาที่สอนทั้งหมด)/.test(q);
    const asksGroupsList = /(มีกลุ่มอะไรบ้าง|กลุ่มอะไรบ้าง|กลุ่มทั้งหมด|ทุกกลุ่ม|รายชื่อกลุ่ม|กลุ่มที่สอนทั้งหมด)/.test(q);
    const asksRoomsList = /(มีห้องอะไรบ้าง|ห้องไหนบ้าง|ใช้ห้องอะไร|ใช้ห้องไหน|ห้องที่สอนทั้งหมด)/.test(q);
    const asksTypes = /(ทฤษฎี|ปฏิบัติ|ภาคทฤษฎี|ภาคปฏิบัติ)/.test(q);
    const asksHowMany = /(มีกี่|จำนวน|กี่รายการ|กี่วิชา|กี่กลุ่ม|กี่ห้อง|กี่ครั้ง)/.test(q);

    if (!subject && asksSubjectsList && !day) {
        const subjects = uniqueBy(allItems.filter(x => x.code || x.subject), x => x.code || normalize(x.subject));
        return [
            '📚 รายวิชาที่มีในตาราง',
            '',
            ...subjects.map((x, i) => {
                const occurrences = allItems.filter(y => (x.code && y.code === x.code) || normalize(y.subject) === normalize(x.subject));
                const groups = uniqueBy(occurrences.map(y => y.group).filter(Boolean));
                return (i + 1) + '. ' + (x.subject || '-') + ' (' + (x.code || '-') + ') — ' + occurrences.length + ' คาบ / ' + groups.length + ' กลุ่ม';
            })
        ].join('\n');
    }

    if (!findGroup(q) && asksGroupsList) {
        const groups = uniqueBy(allItems.map(x => x.group).filter(Boolean));
        return [
            '👥 กลุ่มที่พบในตารางทั้งหมด',
            '',
            ...groups.map((g, i) => {
                const items = groupItems(g);
                const subjects = uniqueBy(items.map(x => x.subject).filter(Boolean));
                return (i + 1) + '. ' + g + ' — ' + subjects.length + ' วิชา / ' + items.length + ' คาบ';
            })
        ].join('\n');
    }

    if (asksRoomsList) {
        const rooms = uniqueBy(allItems.map(x => x.room).filter(Boolean));
        return [
            '🏫 ห้องที่ใช้สอน',
            '',
            ...rooms.map((room, i) => {
                const items = allItems.filter(x => x.room === room);
                return (i + 1) + '. ' + room + ' — ' + items.length + ' คาบ';
            })
        ].join('\n');
    }

    if (asksTypes && asksHowMany) {
        const practical = allItems.filter(x => /ปฏิบัติ/.test(x.type || '')).length;
        const theory = allItems.filter(x => /ทฤษฎี/.test(x.type || '')).length;
        return [
            '📊 แยกตามประเภทการสอน',
            '',
            '• ปฏิบัติ — ' + practical + ' คาบ',
            '• ทฤษฎี — ' + theory + ' คาบ',
            '• รวม — ' + allItems.length + ' คาบ'
        ].join('\n');
    }

    if (subject) {
        const occurrences = subjectItems(subject);
        const uniqueDays = uniqueBy(occurrences.map(x => x.day));
        const uniqueGroups = uniqueBy(occurrences.map(x => x.group).filter(Boolean));
        const totalMinutes = minutesForClasses(occurrences);

        if (/(กี่คาบ|กี่ครั้ง|จำนวนคาบ|สอนกี่ครั้ง)/.test(q)) {
            return [
                '📘 ' + subject.subject,
                '',
                'สอนทั้งหมด ' + occurrences.length + ' คาบ',
                'เวลาสอนรวม ' + formatDuration(totalMinutes),
                'สอน ' + uniqueDays.length + ' วัน',
                '',
                ...occurrences.map(c => '• วัน' + DAY_LABELS[c.day] + ' ' + c.time + ' — ' + (c.group || '-'))
            ].join('\n');
        }

        if (/(กลุ่มไหน|กลุ่มอะไร|กลุ่มใด|มีกลุ่ม)/.test(q)) {
            return [
                '📘 ' + subject.subject,
                '',
                'กลุ่มที่เรียน:',
                ...uniqueGroups.map(g => '• ' + g)
            ].join('\n');
        }

        if (/(วันไหน|เมื่อไหร่|ตอนไหน|กี่โมง|เวลา|สอนวัน|เรียนวัน)/.test(q)) {
            const filtered = day ? occurrences.filter(c => c.day === day) : occurrences;
            return filtered.length
                ? filtered.map(c => '• วัน' + DAY_LABELS[c.day] + ' | ' + formatClassComplete(c)).join('\n')
                : '- ไม่พบวิชานี้ในวัน' + DAY_LABELS[day];
        }

        if (/(ห้อง|เรียนที่ไหน|อยู่ไหน)/.test(q)) {
            const rooms = uniqueBy(occurrences.map(c => c.room).filter(Boolean));
            return '🏫 ' + subject.subject + ' สอนที่: ' + rooms.join(', ');
        }

        return occurrences.length
            ? ['📘 ' + subject.subject + ' (' + subject.code + ')', '', ...occurrences.map(c => '• วัน' + DAY_LABELS[c.day] + ' ' + formatClass(c))].join('\n')
            : '- ไม่พบวิชานี้ในตาราง';
    }

    if (subject) {
        const occurrences = Object.entries(schedule).flatMap(([d, classes]) => classes.filter(c => c.code === subject.code || normalize(c.subject) === normalize(subject.subject)).map(c => ({ day: d, ...c })));
        if (day) { const dayOccurrences = occurrences.filter(c => c.day === day); if (!dayOccurrences.length) return '- วัน' + DAY_LABELS[day] + 'ไม่มีวิชา' + subject.subject; return ['วิชา ' + subject.subject + ' (' + subject.code + ') วัน' + DAY_LABELS[day], ...dayOccurrences.map(formatClassComplete)].join('\n'); }
        if (/(วันไหน|เมื่อไหร่|ตอนไหน|กี่โมง|เวลา|สอนวัน|เรียนวัน)/.test(q)) return occurrences.map(c => 'วัน' + DAY_LABELS[c.day] + ' | ' + formatClassComplete(c)).join('\n') || '- ไม่พบวิชานี้ในตาราง';
        if (/(ห้อง|เรียนที่ไหน|อยู่ไหน)/.test(q)) return occurrences.map(c => '- วัน' + DAY_LABELS[c.day] + ' ' + (c.time || '-') + ' ห้อง ' + (c.room || '-')).join('\n') || '- ไม่พบวิชานี้ในตาราง';
        return occurrences.map(c => '- วัน' + DAY_LABELS[c.day] + ' ' + formatClass(c)).join('\n') || '- ไม่พบวิชานี้ในตาราง';
    }

    if (day && (scheduleWords.test(q) || explicitDay || contextDay)) {
        let classes = getDayClasses(day);
        if (/(เช้า|ตอนเช้า)/.test(q)) classes = classes.filter(c => (parseTimeRange(c.time)?.start ?? 9999) < 12 * 60);
        if (/(บ่าย|ตอนบ่าย)/.test(q)) classes = classes.filter(c => (parseTimeRange(c.time)?.start ?? -1) >= 12 * 60);
        if (/(เย็น|ตอนเย็น)/.test(q)) classes = classes.filter(c => (parseTimeRange(c.time)?.start ?? -1) >= 17 * 60);
        if (!classes.length) return '- วัน' + DAY_LABELS[day] + 'ไม่มีคาบตามช่วงเวลาที่ถามครับ';
        if (/(คาบแรก|แรกสุด|เริ่มกี่โมง|เริ่มเรียน)/.test(q)) return '- คาบแรกวัน' + DAY_LABELS[day] + 'เริ่ม ' + classes[0].time;
        if (/(คาบสุดท้าย|สุดท้าย|เลิกกี่โมง|เลิกเรียน)/.test(q)) return '- คาบสุดท้ายวัน' + DAY_LABELS[day] + 'คือ ' + classes[classes.length - 1].subject + ' (' + classes[classes.length - 1].time + ')';
        if (/(กี่คาบ|กี่วิชา|กี่ครั้ง)/.test(q)) return '- วัน' + DAY_LABELS[day] + 'มี ' + classes.length + ' คาบ';
        if (/(ว่าง|มีช่วงว่าง|พัก|ไม่มีเรียน)/.test(q)) return ['- วัน' + DAY_LABELS[day] + 'ช่วงที่ว่าง', ...getFreeSlots(getDayClasses(day)).map(s => '- ' + s)].join('\n');
        return ['ตารางวัน' + DAY_LABELS[day], ...classes.map(formatClassComplete)].join('\n');
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
    const localAnswer = localScheduleAnswer(userMessage, history);
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

    try {
        // ใช้ non-stream เพื่อป้องกันกรณี provider ส่ง stream ว่าง
        // และลอง fallback model อัตโนมัติหาก model/provider ตัวแรกไม่ตอบ
        const candidates = [...new Set([resolvedModel, ...MODEL_CANDIDATES])];
        let completion = null;
        let lastError = null;

        for (const candidate of candidates) {
            try {
                const result = await hf.chatCompletion({
                    model: candidate,
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

                const answer = result?.choices?.[0]?.message?.content?.trim();
                if (answer) {
                    completion = answer;
                    resolvedModel = candidate;
                    console.log('HF response model:', resolvedModel, '| provider:', resolvedProvider);
                    break;
                }

                lastError = new Error('Hugging Face returned an empty response for ' + candidate);
                console.warn(lastError.message);
            } catch (error) {
                lastError = error;
                console.warn('HF model failed:', candidate, '|', error.message);
            }
        }

        if (!completion) {
            throw lastError || new Error('Hugging Face returned an empty response');
        }

        const fullAnswer = completion;
        res.write(fullAnswer);

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

(async () => {
    await resolveAvailableModel();
    app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
        console.log(`HF model: ${resolvedModel} | provider: ${resolvedProvider}`);
    });
})();
