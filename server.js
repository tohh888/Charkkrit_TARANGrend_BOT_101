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
            model: "Qwen/Qwen2.5-72B-Instruct",
            temperature: 0.1,
            messages: [
                {
                    role: "system",
                    content: `คุณคือ AI ผู้ช่วยตอบตารางสอนของอาจารย์จักรกฤษณ์ วงศ์อาษา 
ข้อมูลทั้งหมดอยู่ใน JSON นี้:
${JSON.stringify(scheduleData)}

แนวทางการตอบคำถาม (เคร่งครัดมาก):
1. ห้ามเขียนตอบเป็นความเรียงยาวๆ ติดกันเด็ดขาด ให้ตอบแยกบรรทัดเป็นหัวข้ออย่างเป็นระเบียบ
2. เมื่อตอบข้อมูลทั่วไป ให้จัดโครงสร้างดังนี้:
   • ข้อมูลอาจารย์:
     - ชื่อ: นายจักรกฤษณ์ วงศ์อาษา
     - วุฒิการศึกษา: [ตอบตาม JSON]
     - ตำแหน่ง: [ตอบตาม JSON]
     - สังกัด: [ตอบตาม JSON]
   • ข้อมูลภาคการศึกษา:
     - ภาคเรียน: [ตอบตาม JSON เช่น 1/2569]
3. หากผู้ใช้ถามถึงเวลาหลังจบคาบเรียนสุดท้ายของวัน ให้ตอบว่า "โปรดติดต่อกับครูผู้สอนโดยตรง"
4. ตอบเฉพาะภาษาไทย สั้น กระชับ ตรงประเด็น ห้ามใส่วงเล็บแก้ไขตัวเองเด็ดขาด`
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
