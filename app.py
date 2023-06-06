from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-squad")
model = AutoModelForQuestionAnswering.from_pretrained("savasy/bert-base-turkish-squad")
nlp=pipeline("question-answering", model=model, tokenizer=tokenizer)

@app.route('/question', methods=['POST'])
def question():
    data = request.get_json()
    if 'question' in data and 'context' in data : 
        res = nlp(data['question'], data['context'])
        return jsonify(res)
    else :
        return abort(400)
    
if __name__ == '__main__' : 
    app.run()