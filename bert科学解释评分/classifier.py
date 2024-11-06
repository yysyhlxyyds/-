#定义下游任务模型

from getbert import *
import torch

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                 attention_mask=attention_mask)

        out = self.fc(out.last_hidden_state[:, 0])

        out = out.softmax(dim=1)

        return out

classifier = Classifier()

classifier(input_ids=input_ids_train,
                 attention_mask=attention_mask_train)