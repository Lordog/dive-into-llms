# การปรับจูนและปรับใช้โมเดลภาษาที่ผ่านการฝึกล่วงหน้า (Practical Guide)

สรุป: บทนี้สอนวิธีเลือกโมเดลที่เหมาะสม ปรับจูน (fine-tune) ให้เหมาะกับงานเฉพาะ และปรับใช้เป็นเดโมผ่าน Gradio Spaces

หัวข้อสำคัญ
- เป้าหมาย: เรียนใช้ Transformers, ทำ fine-tuning, สร้าง demo ด้วย Gradio, รู้จักการเลือกโมเดล
- ตัวอย่างงาน: การจัดประเภทข้อความ (เช่น ตรวจจับข่าวปลอม)

1) เตรียมสภาพแวดล้อม
- แหล่งอ้างอิง: https://github.com/huggingface/transformers
- แนะนำสภาพแวดล้อม (ตัวอย่าง):
  - conda create -n llm python=3.9
  - conda activate llm
  - pip install transformers
  - pip install -r requirements.txt (ลบ torch ที่ต้องการติดตั้งอัตโนมัติออกก่อนถ้าต้องการ)
- หากดาวน์โหลดช้า: ใช้ -i https://pypi.tuna.tsinghua.edu.cn/simple
- ติดตั้ง PyTorch แนะนำด้วย conda: conda install pytorch

2) เตรียมข้อมูล
- ตัวอย่าง dataset: Kaggle "nlp-getting-started" (ทวิตเตอร์ตรวจจับข่าวปลอม)
  https://www.kaggle.com/c/nlp-getting-started/data

3) โครงสร้างตัวอย่าง (เวอร์ชันแยกส่วนสำหรับเรียนรู้)
- ไฟล์หลัก: main.py (รัน), utils_data.py (โหลดข้อมูล), modeling_bert.py (โมเดล)
- รันฝึก-ประเมิน-คาดการณ์:
  ```bash
  python main.py
  ```

4) ปรับจูนด้วยสคริปต์รวม (run_classification.py)
- ตัวอย่างการรัน:
  ```bash
  python run_classification.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/train.csv \
    --validation_file data/val.csv \
    --test_file data/test.csv \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name "text" \
    --text_column_delimiter "\n" \
    --label_column_name "target" \
    --do_train --do_eval --do_predict \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir experiments/
  ```
- ปัญหาทั่วไป:
  - “Network is unreachable” → ดาวน์โหลดโมเดลด้วยมือจาก Hugging Face
  - ปัญหาการโหลด metric (evaluate) → ดาวน์โหลด evaluate จาก GitHub และชี้ --metric_name ไปยังไฟล์ .py ในเครื่อง

5) ปรับใช้เป็นเดโมบน Gradio Spaces
- เอกสาร: https://huggingface.co/docs/hub/en/spaces-sdks-gradio
- สร้าง Space: https://huggingface.co/new-space?sdk=gradio
- อัปโหลดไฟล์หลัก (app.py), requirements.txt และโมเดล
- ตัวอย่าง requirements:
  ```
  transformers==4.30.2
  torch==2.0.0
  ```

6) ข้อเสนอแนะเชิงปฏิบัติ
- ทดลองงานอื่น ๆ เช่น sentiment analysis, news classification
- ลองโมเดลอื่น (T5, ELECTRA) และเทคนิคเช่น LoRA สำหรับ fine-tuning เบาๆ

แหล่งข้อมูลเพิ่มเติม
- Hugging Face Transformers docs (จีน/อังกฤษ): https://huggingface.co/docs/transformers
- Llama2 docs: https://huggingface.co/docs/transformers/en/model_doc/llama2
- LoRA example: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/LoRA_Tuning_PEFT.ipynb