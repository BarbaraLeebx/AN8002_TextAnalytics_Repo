# %%
from flask import Flask, request, render_template
from textblob import TextBlob
from transformers import pipeline

# %%
app = Flask(__name__)

# %%
transformersPipeline = pipeline('sentiment-analysis', model="mrm8488/bert-small-finetuned-squadv2",
                                tokenizer="mrm8488/bert-small-finetuned-squadv2")


# %%
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        print(text)
        r1 = TextBlob(text).sentiment
        r2 = transformersPipeline(text)
        return(render_template("index.html", result1=r1, result2=r2))
    else:
        return(render_template("index.html", result1=" ", result2=" "))


# %%
if __name__ == "__main__":  # to ensure that it is this program running in the cloud
    app.run()


# %%
