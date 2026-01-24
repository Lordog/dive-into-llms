# การปรับจูนและปรับใช้โมเดลภาษาที่ผ่านการฝึกล่วงหน้า

> คำแนะนำ: ส่วนนี้แนะนำการปรับจูนโมเดลที่ผ่านการฝึกล่วงหน้า  
ต้องการเพิ่มประสิทธิภาพของโมเดลที่ผ่านการฝึกล่วงหน้าให้กับงานเฉพาะหรือไม่? มาเลือกโมเดลที่เหมาะสม ปรับจูนบนงานเฉพาะ แล้วนำโมเดลที่ปรับจูนแล้วไปปรับใช้เป็นเดโมที่ใช้งานได้!

## เป้าหมายของบทเรียนนี้:
1. คุ้นเคยกับการใช้ไลบรารี Transformers
2. เข้าใจการปรับจูน (fine-tuning) และการอนุมานของโมเดลที่ผ่านการฝึกล่วงหน้า (ทั้งเวอร์ชันแยกส่วนเพื่อตั้งค่าได้ และเวอร์ชันรวมสำเร็จรูป)
3. สามารถใช้ Gradio Spaces ในการปรับใช้เป็นเดโม
4. เข้าใจการเลือกประเภทโมเดลต่างๆ และกรณีการใช้งาน

## เนื้อหาของบทเรียนนี้:
### 1. การเตรียมงาน:
#### 1.1 ทำความรู้จักไลบรารี: Transformers
https://github.com/huggingface/transformers

> 🤗 Transformers ให้ API และเครื่องมือที่ช่วยดาวน์โหลดและฝึกโมเดลสมัยใหม่ได้อย่างสะดวก การใช้โมเดลที่ผ่านการฝึกล่วงหน้าช่วยลดการใช้คำนวณและคาร์บอนฟุตพริ้นต์ อีกทั้งประหยัดเวลาเมื่อเทียบกับการฝึกจากศูนย์
- การประมวลผลภาษาธรรมชาติ (NLP): การจัดประเภทข้อความ, การจดจำหน่วยชื่อเฉพาะ, การตอบคำถาม, การทำ language modeling, สรุปข้อความ, การแปล, การเลือกแบบฝึกหัด และการสร้างข้อความ
- วิชัน: การจัดประเภทภาพ, ตรวจจับวัตถุ, แยกส่วนเชิงความหมาย
- เสียง: การรู้จำเสียงอัตโนมัติและการจัดประเภทเสียง
- มัลติ-โมดอล: Q&A บนตาราง, OCR, การดึงข้อมูลจากเอกสารสแกน, การจัดประเภทวิดีโอ และ Visual Question Answering

เอกสารภาษาไทย/จีน: https://huggingface.co/docs/transformers/main/zh/index

![huggingface](./assets/huggingface.PNG)

#### 1.2 ติดตั้งสภาพแวดล้อม: ยกตัวอย่างการจัดหมวดข้อความ (เช่น ตรวจจับข่าวปลอม)
1. เข้าไปที่ตัวอย่างงาน text-classification ของ Transformers เพื่อดู README และดาวน์โหลด requirements.txt กับ run_classification.py
https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification

2. ติดตั้งสภาพแวดล้อม:
- สร้าง conda env: conda create -n llm python=3.9
- เข้า virtual env: conda activate llm
- pip install transformers
- ลบ torch ที่ requirements จะติดตั้งอัตโนมัติ แล้วติดตั้งด้วย pip install -r requirements.txt

> ถ้าดาวน์โหลดช้า ให้ใช้ mirror ในประเทศ: pip [Packages] -i https://pypi.tuna.tsinghua.edu.cn/simple

> หากใช้ mirror ในประเทศเพื่อติดตั้ง PyTorch บางครั้งจะได้เฉพาะเวอร์ชัน CPU และไม่สามารถใช้ GPU ได้ ดังนั้น—
- ใช้ conda install pytorch

> หากดาวน์โหลดช้า ให้ดูวิธีตั้งค่า conda mirror ตามบล็อก: https://blog.csdn.net/weixin_42797483/article/details/132048218

3. เตรียมข้อมูล: ในที่นี้ใช้ชุดข้อมูลทวิตเตอร์ข่าวปลอมจาก Kaggle เป็นตัวอย่าง: https://www.kaggle.com/c/nlp-getting-started/data

#### 1.3 ชุดโค้ดตัวอย่างที่จัดเตรียมไว้ (โค้ดเดโมและข้อมูล)
(1) เวอร์ชันแยกส่วนที่ปรับแต่งได้ (โมดูลหลักแยกชัดเจน เหมาะสำหรับการเรียนรู้และปรับแต่งการโหลดข้อมูล โครงสร้างโมเดล ตัวชี้วัด ฯลฯ)
- TextClassificationCustom ดาวน์โหลด: https://pan.quark.cn/s/00dae5c2b128

(2) เวอร์ชันรวมสำเร็จ (โค้ดใหญ่ขึ้น ใช้การเรียกพารามิเตอร์สำเร็จรูป เหมาะสำหรับการรันทดสอบโดยตรง)
- TextClassification ดาวน์โหลด: https://pan.quark.cn/s/9d0510f1c98d

### 2. พัฒนาแบบกำหนดได้บนเวอร์ชันแยกส่วน (MVP)
มี 3 ไฟล์หลัก: main.py (โปรแกรมหลัก), utils_data.py (โหลดและจัดการข้อมูล), modeling_bert.py (โครงสร้างโมเดล)

![project structure](./assets/0.png)

#### 2.1 ทำความเข้าใจโมดูลสำคัญ
1. โหลดและประมวลผลข้อมูล (utils_data.py)
![utils_data.py](./assets/1.png)

2. โหลดโมเดล (modeling_bert.py)
![modeling_bert_1.py](./assets/2.png)
![modeling_bert_2.py](./assets/3.png)

3. ฝึก/ตรวจสอบ/คาดการณ์ (main.py)
![main.py](./assets/4.png)

#### 2.2 รันฝึก/ตรวจสอบ/คาดการณ์แบบครบวงจร
```shell
python main.py
```

### 3. ปรับจูนบนเวอร์ชันรวม (Optional — ใช้ run_classification.py)
#### 3.1 ทำความเข้าใจโมดูลสำคัญ:
1. โหลดข้อมูล (csv หรือ json)
![load data](./assets/5.png)

2. ประมวลผลข้อมูล
![process data](./assets/6.png)

3. โหลดโมเดล
![load model](./assets/7.png)

4. ฝึก/ตรวจสอบ/คาดการณ์
![train dev predict](./assets/8.png)

#### 3.2 ฝึกโมเดล
ทำการตรวจสอบบนชุดพัฒนา และคาดการณ์บนชุดทดสอบ โดยเรียกสคริปต์ดังนี้:
```shell
python run_classification.py \
    --model_name_or_path  bert-base-uncased \
    --train_file data/train.csv \
    --validation_file data/val.csv \
    --test_file data/test.csv \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name "text" \
    --text_column_delimiter "\n" \
    --label_column_name "target" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir experiments/
```

ถ้าเกิด error หรือกระบุก ควรตรวจสอบเครือข่าย:
1. หากดาวน์โหลดโมเดลแล้วขึ้น “Network is unreachable” ให้ดาวน์โหลดโมเดลด้วยมือจาก: https://huggingface.co/google-bert/bert-base-uncased  
2. หากหลังจากใส่ข้อมูลแล้วการประมวลผลค้าง เมื่อกด CTRL+C แล้วเห็นว่าค้างที่ “connection” ให้ลองดูที่การโหลดแพ็กเกจ evaluate ซึ่งอาจพยายามเชื่อมต่อเครือข่ายแล้วล้มเหลว

![bug](./assets/9.png)

ในกรณีนี้ สามารถดาวน์โหลดแพ็กเกจ evaluate จาก GitHub: https://github.com/huggingface/evaluate/tree/main แล้วเปลี่ยนพารามิเตอร์ --metric_name เป็นเส้นทางไฟล์ของมาตรวัดบนเครื่อง เช่น:
```shell
python run_classification.py \
    --model_name_or_path  bert-base-uncased \
    --train_file data/train.csv \
    --validation_file data/val.csv \
    --test_file data/test.csv \
    --shuffle_train_dataset \
    --metric_name evaluate/metrics/accuracy/accuracy.py \
    --text_column_name "text" \
    --text_column_delimiter "\n" \
    --label_column_name "target" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir experiments/
```

![reference result](./assets/10.png)

### 4. การปรับใช้โมเดล: หลังฝึกเสร็จ เราสามารถสร้างเดโมออนไลน์บน Gradio Spaces
#### 4.1 เอกสาร Gradio Spaces
https://huggingface.co/docs/hub/en/spaces-sdks-gradio

#### 4.2 สร้าง Spaces
1. https://huggingface.co/new-space?sdk=gradio  
2. หมายเหตุ: หากเปิดไม่ได้ ลองใช้การเชื่อมต่อแบบมี proxy หรือ VPN

![Gradio Spaces](./assets/gradio.png)

#### 4.3 โค้ดอนุมานสำคัญ
ดูในไฟล์ app.py ของแพ็กเกจตัวอย่าง

![app.py](./assets/11.png)

#### 4.4 อัปโหลด app.py, ไฟล์สภาพแวดล้อม และโมเดลไปยัง Gradio Spaces
1. ไฟล์กำหนดสภาพแวดล้อม (requirements.txt)
```
transformers==4.30.2
torch==2.0.0
```

2. โครงสร้างไฟล์ตัวอย่าง
![file overview](./assets/12.png)

3. ตัวอย่างผลลัพธ์เดโมที่ประสบความสำเร็จ: https://huggingface.co/spaces/cooelf/text-classification  
ในมุมขวาบนที่แท็บ “Files” สามารถดูโค้ดได้

![Files](./assets/13.png)

4. ไข่อีสเตอร์: บนแพลตฟอร์ม Spaces สามารถดูเดโมยอดนิยมรายสัปดาห์และค้นหาเดโม/โมเดลที่สนใจเพื่อทดลองใช้งาน
![Spaces](./assets/14.png)

### 5. แบบฝึกหัดเชิงลึก
1. ทดลองงานจัดประเภท/ถดถอยอื่น ๆ เช่น การวิเคราะห์อารมณ์, การจัดหมวดข่าว, การจัดหมวดช่องโหว่ ฯลฯ  
2. ทดลองโมเดลประเภทอื่น ๆ เช่น T5, ELECTRA ฯลฯ

### 6. โมเดลที่ใช้กันบ่อยอื่น ๆ
1. แบบจำลองตอบคำถาม (Question Answering): https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering  
2. การสรุปข้อความ (Summarization): https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization  
3. เรียกใช้ Llama2 เพื่ออนุมาน: https://huggingface.co/docs/transformers/en/model_doc/llama2  
4. ปรับจูน Llama2 แบบน้ำหนักเบา (LoRA): https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/LoRA_Tuning_PEFT.ipynb

### 7. อ่านเพิ่มเติม
1. บทความรีวิวเชิงลึก 43 หน้าสำหรับ LLMs: [Large Language Models: A Survey] (ลิงก์ไปยัง arXiv ในต้นฉบับ) — (ตัวอย่างต้นฉบับในจีน/ลิงก์)  
   ลิงก์บทความ: https://arxiv.org/pdf/2402.06196.pdf
2. วิดีโอ/การแนะนำ GPT, GPT-2, GPT-3: https://www.bilibili.com/video/BV1AF411b7xQ?t=0.0  
3. วิดีโอ/การแนะนำ InstructGPT: https://www.bilibili.com/video/BV1hd4y187CR?t=0.4
