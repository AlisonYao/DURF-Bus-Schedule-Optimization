######################################################################
# zhuang@x86_64-apple-darwin13 webapp % FLASK_APP=init.py flask run
######################################################################

from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    # demand = request.form["demand"]
    # print(f"%%%%%%% demand: {demand}")
    print("HERE")
    return render_template("index.html")


@app.route("/GA", methods=["GET", "POST"])
def GA():
    demand = request.form["demand"]
    print(f"######## demand: {demand}")

    def somethingComplicated(x):
        return int(x) ** 2

    return render_template("GA.html", demand=somethingComplicated(demand))


if __name__ == "__main__":
    app.run("127.0.0.1", 5000, debug=True)
