# 第一回B3向けの資料
uvを前提としているので、uvじゃなかったらわかりません。
## 使い方
pyproject.tomlに、必要なライブラリが全て指定されているので、
bashなどでこのREADME.mdと同じ階層のディレクトリに移動した後に
```text
uv sync
```
を実行すると、.venvディレクトリが生成されて、全ての必要なライブラリがインストールされます。

さらに、
```text
. .venv/bin/activate
```
を実行して、プロジェクトの環境をアクティベートします。wsl2のユーザー名の隣にディレクトリ名が出てきたら成功です。
エラーが出たら、頑張って動くようにしてください。

.pyファイルを実行したいときは、環境をアクティベートしたあとに
```text
python 任意のファイル名.py
```
を実行することで動かせます。動かない場合は、~/.venv/bin/pythonを直接指定して動かすか、インタープリターのパスを変更する(ctrl+shift+P→インタープリターを選択→./.venv/bin/pythonを選択)か、で8割くらいは直ります。

また、.ipynbファイルを実行したいときは少し手順が面倒で、ipykernelというライブラリがインストールされているのを確認した後にbashなどで
```text
ipython kernel install --user --name=任意のカーネル名
```
を実行し、カーネルを作成。実行したい.iipynbを開いた後、右上の「カーネルを選択」をクリックし、「Jupyter Kernel」から自分の作成したカーネルを選択する。
作成したカーネルが無い場合、真ん中右上のリロードマークを押すか、VScode自体を再起動すると結構出てくる。

