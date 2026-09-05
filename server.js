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

        // สลับมาใช้ DeepSeek-R1 32B เสถียรสูงและฉลาดมาก
        const stream = hf.chatCompletionStream({
            model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            temperature: 0.1,
            messages: [
                {
                    role: "system",
                    content: `คุณคือ AI ผู้ช่วยตอบตารางสอนของอาจารย์จักรกฤษณ์ วงศ์อาษา 
ข้อมูลทั้งหมดอยู่ใน JSON นี้:
${JSON.stringify(scheduleData)}

กฎการตอบคำถาม (เคร่งครัดมาก):
1. ตอบตรงประเด็น ถามอะไรตอบแค่นั้น ห้ามเอาข้อมูลที่ไม่เกี่ยวข้องมาตอบปนเด็ดขาด
2. หากผู้ใช้ถาม "ขอข้อมูลอาจารย์" ให้ตอบเฉพาะข้อมูลบุคคลของอาจารย์เท่านั้น แยกเป็นข้อๆ ดังนี้:
   * ชื่อ: นายจักรกฤษณ์ วงศ์อาษา
   * วุฒิการศึกษา: [ตอบตาม JSON]
   * ตำแหน่ง: [ตอบตาม JSON]
   * สังกัด: [ตอบตาม JSON]
   (ห้ามใส่ข้อมูลภาคการศึกษา หรือเรื่องตารางเรียนลงมาเด็ดขาด)
3. หากผู้ใช้ถามถึงเวลาหลังจบคาบเรียนสุดท้ายของวัน ให้ตอบว่า "โปรดติดต่อกับครูผู้สอนโดยตรง"
4. ใช้เฉพาะภาษาไทย สั้น กระชับ ห้ามใส่วงเล็บแก้ไขตัวเอง ห้ามเขียนเป็นความเรียงติดกัน`
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
            res.status(500).send("ระบบกำลังหนาแน่น กรุณาลองใหม่อีกครั้งในครู่เดียวครับ");
        } else {
            res.write("\n[ระบบกำลังหนาแน่น กรุณาลองใหม่อีกครั้ง]");
            res.end();
        }
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
