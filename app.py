# _*_ coding: utf-8 _*_
# /usr/bin/env python

"""
Author: Thomas Chen
Email: guyanf@gmail.com
Company: Thomas

date: 2025/1/21 10:18
desc:
"""

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    title = "GDP by Country (1960-2023)"
    return render_template('index.html', title=title)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
