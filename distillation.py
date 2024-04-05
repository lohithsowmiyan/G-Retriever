from transformers import (
    #GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    #GPT2LMHeadModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Subset
from random import sample
from torch.utils.data import DataLoader
from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.collate import collate_fn

from pathlib import Path
import wandb

#############
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 128

TEMPERATURE = 2.0
ALPHA = 0.5
#############

IGNORE_INDEX = -100


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'


wandb_log = True

args = parse_args_llama()
dataset = load_dataset[args.dataset]()
idx_split = dataset.get_idx_split()

max_new_tokens = args.max_new_tokens
max_txt_len = args.max_txt_len


# in the original code I had random_chunk = False
# random_chunk=True is expected to improve the model performance a bit

teacher_dir1 = llama_model_path[args.llm_model_name]
#teacher_dir2 = PATH / 'models/gpt-705M'

MODEL_NAME = f'Baby-Llama-58M'
MODEL_OUTPUT = Path('./models') /  MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(teacher_dir1)
tokenizer.bos_token = "<s>[INST]"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
tokenizer.padding_side = 'left'

#tokenizer.model_max_length = SEQ_LENGTH

config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    num_hidden_layers=16,
    intermediate_size=1024,
    num_attention_heads=8,
    bos_token_id=tokenizer.convert_tokens_to_ids("<s>[INST]"),
    eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
    pad_token_id= tokenizer.convert_tokens_to_ids("<pad>"),
    max_position_embeddings=2*SEQ_LENGTH,
)

student = LlamaForCausalLM(config)
# student = LlamaForCausalLM.from_pretrained(student_dir)


teacher1 = AutoModelForCausalLM.from_pretrained(teacher_dir1,
 device_map ='cpu')
#teacher2 = GPT2LMHeadModel.from_pretrained(teacher_dir2)
teachers = [teacher1]


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

word_embedding = teacher1.get_input_embeddings()


def convert_row_to_embeddings(samples):
        questions = tokenizer(samples["question"], add_special_tokens=True)
        descriptions = tokenizer(samples["desc"], add_special_tokens=True)
        labels = tokenizer(samples["label"], add_special_tokens=True)

        # encode special tokens
        eos_tokens = tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds =  word_embedding(tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(teacher1.device))
        pad_embeds =  word_embedding(torch.tensor(tokenizer.pad_token_id).to(teacher1.device)).unsqueeze(0)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][: max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][: max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = word_embedding(torch.tensor(input_ids).to(teacher1.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(teacher1.device)
        attention_mask = torch.tensor(batch_attention_mask).to(teacher1.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(teacher1.device)

        #return {'input_embeds' : inputs_embeds, 'attention_mask': attention_mask, 'labels' : label_input_ids}
        return inputs_embeds
    

def convert_row_to_embed2(samples):
        questions = tokenizer(samples["question"], padding='max_length', max_length= SEQ_LENGTH, truncation=True)
        descriptions = tokenizer(samples["desc"],  padding='max_length', max_length= SEQ_LENGTH, truncation=True)

        # encode special tokens
        eos_user_tokens = tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = word_embedding(tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to('cpu'))
        pad_embeds = word_embedding(torch.tensor(tokenizer.pad_token_id).to('cpu')).unsqueeze(0)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[i][:max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = word_embedding(torch.tensor(input_ids).to('cpu'))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to('cpu')
        attention_mask = torch.tensor(batch_attention_mask).to('cpu')

        return inputs_embeds


train_dataset = [dataset[i] for i in idx_split['train']]
train_loader = DataLoader(train_dataset ,batch_size=args.eval_batch_size, drop_last=False,  shuffle=False, collate_fn=collate_fn)

train_dataset = train_loader

embedding_data = []
for _,batch in enumerate(train_dataset):

  embedding_data.append(convert_row_to_embed2(batch))

print("size : ", len(embedding_data))
for e in embedding_data:
  print(e.size())
#print(embedding_data[0].keys())

eval_dataset = [dataset[i] for i in idx_split['val']]
eval_loader = DataLoader(train_dataset, drop_last=False,  shuffle=False, collate_fn=collate_fn)

full_eval_dataset = eval_loader

#EVAL_SAMPLES = len(eval_loader)
#eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
#eval_dataset = Subset(full_eval_dataset, eval_indices)

PATH = Path("./")


print(f'model num parameters: student = {student.num_parameters()}')
print(f'model num parameters: teacher1 = {teacher1.num_parameters()}')
#print(f'model num parameters: teacher2 = {teacher2.num_parameters()}')

#  Distillation Trainer
#  We modified the Trainer from this repo https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker
# to work with an ensemble of teachers 





class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        for teacher in self.teachers:
            # place each teacher on same device as student
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # compute teacher output
        with torch.no_grad():
            all_teacher_logits = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        # assert size
        assert outputs_student.logits.size() == avg_teacher_logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


if wandb_log:
    wandb.login()
    wandb.init(project='babylm', name=MODEL_NAME)


training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    num_train_epochs=6,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=1,  # Set to zero to avoid saving
    report_to="wandb",
    warmup_steps=200, 
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=20,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    weight_decay=0.1,
    alpha=ALPHA,
    temperature=TEMPERATURE,
)

trainer = DistillationTrainer(
        student,
        training_args,
        teacher_models=teachers,
        data_collator=data_collator,
        train_dataset=embedding_data,
        #eval_dataset=eval_dataset,
    )

trainer.train()
#trainer.save_model(MODEL_OUTPUT)
#tokenizer.save_pretrained(MODEL_OUTPUT)











