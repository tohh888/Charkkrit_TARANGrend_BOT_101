require('dotenv').config();
const express = require('express');
const { HfInference } = require('@huggingface/inference');
const scheduleData = require('./schedule.json');

const app = express();
app.use(express.json());
app.use(express.static('public'));

const hf = new HfInference(process.env.HF_TOKEN);

app.post('/api/chat', async (req, res) => {
    console.log("-> ได้รับข้อความจากผู้ใช้:", req.body.message);
    try {
        const userMessage = req.body.message;
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');

        const stream = hf.chatCompletionStream({
            model: "Qwen/Qwen2.5-32B-Instruct",
            temperature: 0.1,
            messages: [
                {
                    role: "system",
                    content: `คุณคือ AI ผู้ช่วยตอบตารางสอนของอาจารย์จักรกฤษณ์ วงศ์อาษา 
ข้อมูลทั้งหมดอยู่ใน JSON นี้:
${JSON.stringify(scheduleData)}

กฎการตอบคำถาม (เคร่งครัดมาก):
1. ตอบเฉพาะคำตอบสุดท้ายเท่านั้น ห้ามอธิบายกระบวนการคิด หรือแสดงข้อความวิเคราะห์ตนเองเด็ดขาด
2. ตอบตรงประเด็น สั้น กระชับ แยกบรรทัดเป็นข้อๆ (Bullet points) เท่านั้น
3. หากผู้ใช้ถาม "ขอข้อมูลอาจารย์" ให้ตอบเฉพาะข้อมูลส่วนบุคคลเป็นข้อๆ ดังนี้:
   * ชื่อ: นายจักรกฤษณ์ วงศ์อาษา
   * วุฒิการศึกษา: [ตอบตาม JSON]
   * ตำแหน่ง: [ตอบตาม JSON]
   * สังกัด: [ตอบตาม JSON]
   (ห้ามใส่ข้อมูลภาคการศึกษา หรือเรื่องตารางเรียนลงมาเด็ดขาด)
4. หากผู้ใช้ถามถึงเวลาหลังจบคาบเรียนสุดท้ายของวัน ให้ตอบว่า "โปรดติดต่อกับครูผู้สอนโดยตรง"`
                },
                { role: "user", content: userMessage }
            ],
            max_tokens: 1000, // <--- ขยายจาก 300 เป็น 1000 เพื่อรองรับตารางสอนยาวๆ
        });

        for await (const chunk of stream) {
            if (chunk.choices[0]?.delta?.content) {
                res.write(chunk.choices[0].delta.content);
            }
        }
        res.end();
        console.log("-> ส่งคำตอบสำเร็จ!");

    } catch (error) {
        console.error("-> เกิดข้อผิดพลาด:", error.message || error);
        if (!res.headersSent) {
            res.status(500).send("ระบบกำลังหนาแน่น กรุณาลองใหม่อีกครั้งในครู่เดียวครับ");
        } else {
            res.write("\n[เกิดข้อผิดพลาดในการประมวลผล]");
            res.end();
        }
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
