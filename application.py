from flask import Flask, render_template, request
from recommender import recommend_nmf

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("main.html", title1="Cascabel")


@app.route('/recommendations')
def recommender():
    user_input_from_app = request.args
    top10_films = recommend_nmf(user_input_from_app)
    return render_template("recommender.html", films_var=top10_films)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
