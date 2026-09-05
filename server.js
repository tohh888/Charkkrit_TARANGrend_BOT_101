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
            temperature: 0.1, // ปรับให้ต่ำมาก เพื่อให้ตอบตรงประเด็นและสั้นที่สุด
            messages: [
                {
                    role: "system",
                    content: `คุณคือ AI ผู้ช่วยตอบตารางสอนของอาจารย์จักรกฤษณ์ วงศ์อาษา 
ข้อมูลทั้งหมดอยู่ใน JSON นี้:
${JSON.stringify(scheduleData)}

แนวทางการตอบคำถาม (เคร่งครัดมาก):
1. ตอบแบบ "ถามมา-ตอบไป" สั้น กระชับ ตรงประเด็น ไม่ต้องมีคำเกริ่นหรือคำลงท้ายยาวๆ
2. ตอบเฉพาะภาษาไทยเท่านั้น ห้ามใส่วงเล็บแก้ไขคำพูด ห้ามใส่หมายเหตุ หรือแสดงขั้นตอนการคิดเด็ดขาด
3. หากผู้ใช้ถามถึงเวลาหรือช่วงเวลาที่ **หลังจบคาบเรียนสุดท้ายของวันนั้นๆ** ให้ตอบกลับว่า "โปรดติดต่อกับครูผู้สอนโดยตรง" เสมอ
4. หากเป็นข้อมูลทั่วไป สรุปเป็นหัวข้อสั้นๆ ให้จบภายใน 2-3 บรรทัด`
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
