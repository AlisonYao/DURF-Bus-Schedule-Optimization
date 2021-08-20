######################################################################
# zhuang@x86_64-apple-darwin13 webapp % FLASK_APP=init.py flask run
######################################################################

from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/GA", methods=["GET", "POST"])
def GA():
    # Jinqiao to AB
    JQ2AB_demand1 = request.form["JQ2AB_demand1"]
    JQ2AB_demand2 = request.form["JQ2AB_demand2"]
    JQ2AB_demand3 = request.form["JQ2AB_demand3"]
    JQ2AB_demand4 = request.form["JQ2AB_demand4"]
    JQ2AB_demand5 = request.form["JQ2AB_demand5"]
    JQ2AB_demand6 = request.form["JQ2AB_demand6"]
    JQ2AB_demand7 = request.form["JQ2AB_demand7"]
    JQ2AB_demand8 = request.form["JQ2AB_demand8"]
    JQ2AB_demand9 = request.form["JQ2AB_demand9"]
    JQ2AB_demand10 = request.form["JQ2AB_demand10"]
    JQ2AB_demand11 = request.form["JQ2AB_demand11"]
    JQ2AB_demand12 = request.form["JQ2AB_demand12"]
    JQ2AB_demand13 = request.form["JQ2AB_demand13"]
    JQ2AB_demand14 = request.form["JQ2AB_demand14"]
    JQ2AB_demand15 = request.form["JQ2AB_demand15"]
    JQ2AB_demand16 = request.form["JQ2AB_demand16"]
    JQ2AB_demand17 = request.form["JQ2AB_demand17"]
    JQ2AB_demand18 = request.form["JQ2AB_demand18"]
    JQ2AB_demand19 = request.form["JQ2AB_demand19"]
    JQ2AB_demand20 = request.form["JQ2AB_demand20"]
    JQ2AB_demand21 = request.form["JQ2AB_demand21"]
    JQ2AB_demand22 = request.form["JQ2AB_demand22"]
    JQ2AB_demand23 = request.form["JQ2AB_demand23"]
    JQ2AB_demand24 = request.form["JQ2AB_demand24"]
    JQ2AB_demand25 = request.form["JQ2AB_demand25"]
    JQ2AB_demand26 = request.form["JQ2AB_demand26"]
    JQ2AB_demand27 = request.form["JQ2AB_demand27"]
    JQ2AB_demand28 = request.form["JQ2AB_demand28"]
    JQ2AB_demand29 = request.form["JQ2AB_demand29"]
    JQ2AB_demand30 = request.form["JQ2AB_demand30"]
    JQ2AB_demand31 = request.form["JQ2AB_demand31"]
    JQ2AB_demand32 = request.form["JQ2AB_demand32"]
    JQ2AB_demand33 = request.form["JQ2AB_demand33"]
    JQ2AB_demand34 = request.form["JQ2AB_demand34"]
    # AB to Jinqiao
    AB2JQ_demand1 = request.form["AB2JQ_demand1"]
    AB2JQ_demand2 = request.form["AB2JQ_demand2"]
    AB2JQ_demand3 = request.form["AB2JQ_demand3"]
    AB2JQ_demand4 = request.form["AB2JQ_demand4"]
    AB2JQ_demand5 = request.form["AB2JQ_demand5"]
    AB2JQ_demand6 = request.form["AB2JQ_demand6"]
    AB2JQ_demand7 = request.form["AB2JQ_demand7"]
    AB2JQ_demand8 = request.form["AB2JQ_demand8"]
    AB2JQ_demand9 = request.form["AB2JQ_demand9"]
    AB2JQ_demand10 = request.form["AB2JQ_demand10"]
    AB2JQ_demand11 = request.form["AB2JQ_demand11"]
    AB2JQ_demand12 = request.form["AB2JQ_demand12"]
    AB2JQ_demand13 = request.form["AB2JQ_demand13"]
    AB2JQ_demand14 = request.form["AB2JQ_demand14"]
    AB2JQ_demand15 = request.form["AB2JQ_demand15"]
    AB2JQ_demand16 = request.form["AB2JQ_demand16"]
    AB2JQ_demand17 = request.form["AB2JQ_demand17"]
    AB2JQ_demand18 = request.form["AB2JQ_demand18"]
    AB2JQ_demand19 = request.form["AB2JQ_demand19"]
    AB2JQ_demand20 = request.form["AB2JQ_demand20"]
    AB2JQ_demand21 = request.form["AB2JQ_demand21"]
    AB2JQ_demand22 = request.form["AB2JQ_demand22"]
    AB2JQ_demand23 = request.form["AB2JQ_demand23"]
    AB2JQ_demand24 = request.form["AB2JQ_demand24"]
    AB2JQ_demand25 = request.form["AB2JQ_demand25"]
    AB2JQ_demand26 = request.form["AB2JQ_demand26"]
    AB2JQ_demand27 = request.form["AB2JQ_demand27"]
    AB2JQ_demand28 = request.form["AB2JQ_demand28"]
    AB2JQ_demand29 = request.form["AB2JQ_demand29"]
    AB2JQ_demand30 = request.form["AB2JQ_demand30"]
    AB2JQ_demand31 = request.form["AB2JQ_demand31"]
    AB2JQ_demand32 = request.form["AB2JQ_demand32"]
    AB2JQ_demand33 = request.form["AB2JQ_demand33"]
    AB2JQ_demand34 = request.form["AB2JQ_demand34"]

    def somethingComplicated(x):
        return int(x) ** 2

    return render_template("GA.html", demand=somethingComplicated(demand1))


if __name__ == "__main__":
    app.run("127.0.0.1", 5000, debug=True)
