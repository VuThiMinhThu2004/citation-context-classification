import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from sklearn.metrics import f1_score

# Đường dẫn đến model đã lưu và tokenizer
model_path = '/content/drive/MyDrive/best_model.pt'
model_name = 'allenai/scibert_scivocab_uncased'
drop_out = 0.1

LMTokenizer = AutoTokenizer.from_pretrained(model_name)
LMModel = AutoModel.from_pretrained(model_name)

device = 'cuda' if cuda.is_available() else 'cpu'

# Định nghĩa lớp mô hình
class LMClass(torch.nn.Module):
    def __init__(self):
        super(LMClass, self).__init__()
        self.l1 = LMModel
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(drop_out)
        self.classifier = torch.nn.Linear(768, 6)

    def forward(self, data):
        input_ids_cgt = data['CGT_ids'].to(device, dtype=torch.long)
        attention_mask_cgt = data['CGT_mask'].to(device, dtype=torch.long)

        input_ids_cdt = data['CDT_ids'].to(device, dtype=torch.long)
        attention_mask_cdt = data['CDT_mask'].to(device, dtype=torch.long)

        input_ids_cc = data['CC_ids'].to(device, dtype=torch.long)
        attention_mask_cc = data['CC_mask'].to(device, dtype=torch.long)

        output_cgt = self.l1(input_ids=input_ids_cgt, attention_mask=attention_mask_cgt)
        output_cdt = self.l1(input_ids=input_ids_cdt, attention_mask=attention_mask_cdt)
        output_cc = self.l1(input_ids=input_ids_cc, attention_mask=attention_mask_cc)

        # Lấy hidden state của các đầu vào
        hidden_state_cgt = output_cgt[0][:, 0]
        hidden_state_cdt = output_cdt[0][:, 0]
        hidden_state_cc = output_cc[0][:, 0]

        # Tổng hợp lại các đặc trưng từ CGT, CDT, CC
        combined_hidden = hidden_state_cgt + hidden_state_cdt + hidden_state_cc

        pooler = self.pre_classifier(combined_hidden)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Khởi tạo lớp dự đoán
class LMClassPredictor:
    def __init__(self, model_path, tokenizer, device='cpu'):
        self.device = device
        self.model = LMClass()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer

    def predict(self, cgt, cdt, cc):
        inputs_cgt = self.tokenizer.encode_plus(
            cgt,
            None,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        inputs_cdt = self.tokenizer.encode_plus(
            cdt,
            None,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        inputs_cc = self.tokenizer.encode_plus(
            cc,
            None,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )

        data = {
            'CGT_ids': torch.tensor(inputs_cgt['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device),
            'CGT_mask': torch.tensor(inputs_cgt['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.device),
            'CDT_ids': torch.tensor(inputs_cdt['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device),
            'CDT_mask': torch.tensor(inputs_cdt['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.device),
            'CC_ids': torch.tensor(inputs_cc['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device),
            'CC_mask': torch.tensor(inputs_cc['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.device),
        }

        with torch.no_grad():
            outputs = self.model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1).item()

        return predicted_labels, probabilities.squeeze()

# Khởi tạo đối tượng dự đoán
predictor = LMClassPredictor(model_path=model_path, tokenizer=LMTokenizer, device=device)

# Đọc dữ liệu từ tập tin validation
validation_data = pd.read_csv('/content/drive/MyDrive/validation.csv', sep=',', names=['CGT', 'CDT', 'CC'])

# File CSV lưu kết quả
output_file = '/content/drive/MyDrive/predictions.csv'

# Dự đoán và lưu kết quả vào CSV
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['CGT', 'CDT', 'CC', 'Predicted_Label', 'Probabilities']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for index, row in validation_data.iterrows():
        cgt = row['CGT']
        cdt = row['CDT']
        cc = row['CC']
        predicted_label, probabilities = predictor.predict(cgt, cdt, cc)
        writer.writerow({
            'CGT': cgt,
            'CDT': cdt,
            'CC': cc,
            'Predicted_Label': predicted_label,
            'Probabilities': probabilities.tolist()
        })

print(f"Predictions saved to {output_file}")
