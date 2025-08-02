from flask import Flask
from flask import render_template
from main import graph
app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/chat/<query>")
def chatbot(query):
    return graph.invoke({"question": query})["answer"]

if __name__ == '__main__':
    app.run(debug=True)
