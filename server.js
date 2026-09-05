require('dotenv').config();
const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const scheduleData = require('./schedule.json');

const app = express();
app.use(express.json());
app.use(express.static('public'));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.post('/api/chat', async (req, res) => {
    console.log("-> ได้รับข้อความจากผู้ใช้:", req.body.message);
    try {
        const userMessage = req.body.message;
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');

        // ใช้ gemini-1.5-flash รองรับฟรี 1,500 ครั้ง/วัน และ 15 ครั้ง/นาที
        const model = genAI.getGenerativeModel({ 
            model: "gemini-1.5-flash",
            systemInstruction: `คุณคือ AI ผู้ช่วยตอบคำถามตารางสอนของอาจารย์
ข้อมูลทั้งหมดที่คุณต้องใช้อยู่ใน JSON นี้เท่านั้น:
${JSON.stringify(scheduleData)}

คำสั่งในการวิเคราะห์และตอบคำถาม:
1. เรื่องตารางสอนรายวัน: สรุปเวลา, วิชา, ท./ป., ห้อง, และกลุ่มเรียนให้ชัดเจน
2. เรื่องเวลาว่าง: เวลาทำการปกติคือ 08:00 - 17:00 น. ให้เปรียบเทียบเวลาสอนใน JSON แล้วคำนวณช่วงเวลาที่ว่างตอบกลับมาให้ถูกต้อง
3. ข้อมูลอาจารย์: ตอบจาก teacher_info (ชื่อ, วุฒิการศึกษา, ตำแหน่ง, วิทยาลัย ฯลฯ)
4. ตอบด้วยภาษาไทยที่สุภาพ อ่านง่าย กระชับ และตรงประเด็น`
        });

        const result = await model.generateContentStream(userMessage);

        for await (const chunk of result.stream) {
            res.write(chunk.text());
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
