from flask import Flask, render_template
from flask import request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('old_index.html')


def response_page(questions, answers):
    import os
    current_directory = os.getcwd()
    document_path = current_directory + "/response_page.txt"
    result = ""
    for question, answer in zip(questions, answers):
        result += f'<div class="subheading mb-3">Question: {question}</div>\n' \
                  f'<div class="subheading mb-3">Answer :{answer}</div>'
    with open(document_path, "r") as f:
        content = f.read()
        content = content.replace("$$$$$$RESPONSE$$$$$$", result)
    with open(current_directory + "/templates/response.html", "w") as f:
        f.write(content)
    return render_template("response.html")


@app.route('/response/', methods=['POST'])
def my_link():
    paragraph = request.form['paragraph']
    questions = []
    for i in range(10):
        if "question" + str(i) in request.form.to_dict():
            questions.append(request.form.get("question" + str(i)))
    answers = bert_model(paragraph, questions)
    # result = ""
    # for question, answer in zip(questions, answers):
    #     result += f'<div class="subheading mb-3">Question: {question}</div>\n' \
    #               f'<div class="subheading mb-3">Answer :{answer}</div>'
    # return result
    return response_page(questions, answers)


def bert_model(paragraph, questions):
    import tensorflow as tf
    import tensorflow_hub as hub
    from transformers import BertTokenizer


    print(tf.__version__)
    print(hub.__version__)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    questions = questions
    paragraph = paragraph
    answers = []

    for question in questions:
        question_tokens = tokenizer.tokenize(question)
        paragraph_tokens = tokenizer.tokenize(paragraph)
        tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + paragraph_tokens + ['[SEP]']
        input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_word_ids)
        input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)

        input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
        outputs = model([input_word_ids, input_mask, input_type_ids])
        # using `[1:]` will enforce an answer. `outputs[0][0][0]` is the ignored '[CLS]' token logit
        short_start = tf.argmax(outputs[0][0][1:]) + 1
        short_end = tf.argmax(outputs[1][0][1:]) + 1
        answer_tokens = tokens[short_start: short_end + 1]
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
        print(f'Question and answer is : {question}, {answer}')
        answers.append(answer)
    return answers


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, host="0.0.0.0")
