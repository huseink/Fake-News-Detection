import datasets
from nltk.tokenize import word_tokenize


dataset_path = 'predicted_labels2.csv'


results = datasets.load_dataset('csv', data_files=dataset_path, split="train")

rouge = datasets.load_metric("rouge")
meteor = datasets.load_metric("meteor")
bleu = datasets.load_metric("bleu")
sacrebleu = datasets.load_metric("sacrebleu")
bertscore = datasets.load_metric("bertscore")


label = list(results['orj_labl'])
pred_label = list(results['predicted_label'])

tokenized_predictions = [word_tokenize(pred) for pred in pred_label]
tokenized_references = [[word_tokenize(ref)] for ref in label]


score = rouge.compute(predictions=results['predicted_label'],
                      references=results['orj_labl'], rouge_types=["rouge1"])["rouge1"].mid
score_r2 = rouge.compute(predictions=results['predicted_label'],
                         references=results['orj_labl'], rouge_types=["rouge2"])["rouge2"].mid
meteor_score = meteor.compute(predictions=pred_label, references=label)
bleu_score = bleu.compute(
    predictions=tokenized_predictions, references=tokenized_references)
bert_score = bertscore.compute(
    predictions=results['predicted_label'], references=results['orj_labl'], lang='tr')
sacrebleu_score = sacrebleu.compute(
    predictions=tokenized_predictions, references=tokenized_references)
bert_score_array = [round(v, 2) for v in bert_score["f1"]]
bert_score_average = sum(bert_score_array) / len(bert_score_array)
rounded_bert_score_average = round(bert_score_average, 3)
# Define the text file path
txt_file_path = 'generate-extra-makale-koksuz.txt'

# Open the text file in write mode
with open(txt_file_path, 'w', encoding='utf-8') as file:
    file.write('--------------------------\n')
    file.write('Score Evaluations\n')
    file.write('Rouge-1: {}\n'.format(score))
    file.write('Rouge-2: {}\n'.format(score_r2))
    file.write('Meteor: {}\n'.format(round(meteor_score["meteor"], 3)))
    file.write('Bleu: {}\n'.format(round(bleu_score['bleu'], 3)))
    file.write('SacreBleu: {}\n'.format(round(sacrebleu_score['score'], 3)))
    file.write('Bert: {}\n'.format(rounded_bert_score_average))
    file.write('--------------------------')


print('Text file saved successfully.')
