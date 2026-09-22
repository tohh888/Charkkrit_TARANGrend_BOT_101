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

        // สลับมาใช้ Llama-3.3-70B ที่ตอบตรงประเด็น ไม่พ่นกระบวนการคิด
        const stream = hf.chatCompletionStream({
            model: "meta-llama/Llama-3.3-70B-Instruct",
            temperature: 0.1,
            messages: [
                {
                    role: "system",
                    content: `คุณคือ AI ผู้ช่วยตอบตารางสอนของอาจารย์จักรกฤษณ์ วงศ์อาษา 
ข้อมูลทั้งหมดอยู่ใน JSON นี้:
${JSON.stringify(scheduleData)}

กฎการตอบคำถาม (เคร่งครัดมาก):
1. ตอบเฉพาะคำตอบสุดท้ายเท่านั้น ห้ามอธิบายกระบวนการคิด ห้ามอธิบายเหตุผลภาษาอังกฤษ หรือใส่ข้อความวิเคราะห์ตนเองเด็ดขาด
2. ตอบตรงประเด็น สั้น กระชับ เป็นภาษาไทยเท่านั้น
3. หากผู้ใช้ถาม "ขอข้อมูลอาจารย์" ให้ตอบเฉพาะข้อมูลส่วนบุคคลเป็นข้อๆ ดังนี้:
   * ชื่อ: นายจักรกฤษณ์ วงศ์อาษา
   * วุฒิการศึกษา: [ตอบตาม JSON]
   * ตำแหน่ง: [ตอบตาม JSON]
   * สังกัด: [ตอบตาม JSON]
   (ห้ามใส่ข้อมูลภาคการศึกษาลงมาเด็ดขาด)
4. หากผู้ใช้ถามถึงเวลาหรือช่วงเวลาหลังจบคาบเรียนสุดท้ายของวันนั้นๆ ให้ตอบว่า "โปรดติดต่อกับครูผู้สอนโดยตรง"`
                },
                { role: "user", content: userMessage }
            ],
            max_tokens: 300,
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
            res.status(500).send("เกิดข้อผิดพลาดในการเชื่อมต่อ API");
        } else {
            res.write("\n[เกิดข้อผิดพลาดในการประมวลผล]");
            res.end();
        }
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
