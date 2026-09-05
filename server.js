require('dotenv').config();
const express = require('express');
const { HfInference } = require('@huggingface/inference');
const scheduleData = require('./schedule.json');

const app = express();
app.use(express.json());
app.use(express.static('public'));

const hf = new HfInference(process.env.HF_TOKEN);

app.post('/api/chat', async (req, res) => {
    try {
        const userMessage = req.body.message;
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');

        // ใช้โมเดล Open Source ขนาดเล็กที่รองรับภาษาไทยได้ดีผ่าน Hugging Face API
        const stream = hf.chatCompletionStream({
            model: "Qwen/Qwen2.5-72B-Instruct",
            messages: [
                {
                    role: "system",
                    content: `คุณคือ AI ผู้ช่วยตอบคำถามตารางสอนของอาจารย์ ข้อมูลทั้งหมดที่คุณต้องใช้อยู่ใน JSON นี้เท่านั้น: ${JSON.stringify(scheduleData)}`
                },
                { role: "user", content: userMessage }
            ],
            max_tokens: 500,
        });

        for await (const chunk of stream) {
            if (chunk.choices[0]?.delta?.content) {
                res.write(chunk.choices[0].delta.content);
            }
        }
        res.end();
    } catch (error) {
        console.error("Error:", error);
        res.status(500).send("เกิดข้อผิดพลาดในการเชื่อมต่อ API");
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
