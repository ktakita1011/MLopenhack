# 課題 5: クラウドへのデプロイ

## 背景

機械学習モデルを構築するだけでは十分とは言えません。このモデルを公開し、AdventureWorks のチーム メンバー、開発者、外部の開発者が、顧客向けのアプリケーションやサービスに便利な機能を提供できるようにする必要があります。

この課題では、作成したモデルを REST API としてクラウドで運用化 (Web サービスとしてデプロイ) し、AdventureWorks や外部の開発者が、このエンドポイントを使用した一般ユーザー向けアプリケーションを開発して予測を実施できるようにします。

こちらの概念図では、デプロイフェーズがデータ サイエンスのライフサイクルの重要な位置を占めています。

<img src="https://docs.microsoft.com/azure/machine-learning/team-data-science-process/media/overview/tdsp-lifecycle2.png?wt.mc_id=OH-ML-ComputerVision" width="50%" height="50%" alt="Data Science Lifecycle">

## 前提条件

* Docker エンジン (Docker for Windows または Docker for Mac) がインストール済みでローカルまたは VM で実行されていること
* Azure CLI および Azure ML CLI (`azure-cli` パッケージおよび `azure-cli-ml` パッケージ)
* `curl` コマンド ライン ツールや [Postman (英語)](https://www.getpostman.com/) などの、モデルのエンドポイントにRequestを送るツール
* 課題 3 で保存したモデル

## 課題

チーム向けのセットアップ環境でチームの代表者が次の手順を実施します。

* 課題 3 または課題 4 で保存したモデルを、リアルタイム Web サービスとして Azure にデプロイします。 

以下のいずれかのツールを使用して、(行列や json の) データ送信および json 応答の受信を行う API としてモデルをデプロイします。詳しくは[参考資料](#references)をご覧ください。

* Azure Machine Learning CLI 単体
* Azure Machine Learning Workbench
* Docker で Flask を使用するなどの CLI 以外の手法 (詳しくは推奨事項を参照)


AdventureWorks はできるだけシンプルな API を望んでいるため、json でシリアル化された画像を使用します。

## 完了条件

* `curl` または Postman で、URL を使用するかシリアル化して画像をクラウドにデプロイした Web サービスに送り、モデルの出力 (登山用品のクラス) を取得する。

## 参考資料 <a name="references"></a>

**はじめに**

* Azure ML モデルの管理の概要は<a href="https://docs.microsoft.com/ja-jp/azure/machine-learning/desktop-workbench/model-management-overview?wt.mc_id=OH-ML-ComputerVision" target="_blank">こちらのドキュメントを参照</a>
* デプロイメントのガイドは<a href="https://michhar.github.io/deploy-with-azureml-cli-boldly/" target="_blank">こちらを参照 (英語)</a>

**デプロイについての詳細**

* Azure ML Workbench と Azure ML CLI のデプロイに関するマイクロソフトのブログは<a href="https://blogs.technet.microsoft.com/machinelearning/2017/09/25/deploying-machine-learning-models-using-azure-machine-learning/" target="_blank">こちらを参照 (英語)</a>
* Azure ML CLI のデプロイのセットアップ方法は<a href="https://docs.microsoft.com/ja-jp/azure/machine-learning/desktop-workbench/deployment-setup-configuration?wt.mc_id=OH-ML-ComputerVision" target="_blank">こちらのドキュメントを参照</a>
* CLI 以外のデプロイ方法 (AML の代替手段) については<a href="https://github.com/Azure/ACS-Deployment-Tutorial" target="_blank">こちらを参照 (英語)</a>

**Scoringファイルとスキーマの作成に関する資料**

* スキーマ生成の例は<a href="https://docs.microsoft.com/ja-jp/azure/machine-learning/desktop-workbench/model-management-service-deploy?wt.mc_id=OH-ML-ComputerVision#2-create-a-schemajson-file" target="_blank">こちらのドキュメントを参照</a>
* サービスへのデータ入力用の CNTK モデルと `PANDAS` データ型で画像をシリアル化するScoringファイルの例は<a href="https://github.com/Azure/MachineLearningSamples-ImageClassificationUsingCntk/blob/master/scripts/deploymain.py" target="_blank">こちらを参照 (英語)</a>
* サービスへのデータ入力用の `scikit-learn` モデルと`標準的な`データ型 (json) のScoringファイルの例は<a href="https://github.com/Azure/Machine-Learning-Operationalization/blob/master/samples/python/code/newsgroup/score.py" target="_blank">こちらを参照 (英語)</a>
* 上のリンクの手順に従って `run` メソッドと `init` メソッド、およびスキーマ ファイルを作成した後、<a href="https://docs.microsoft.com/ja-jp/azure/machine-learning/desktop-workbench/model-management-service-deploy?wt.mc_id=OH-ML-ComputerVision#4-register-a-model">こちらのドキュメント</a>の「モデルを登録する」の手順を実行
  * マニフェスト作成時に `pip` の requirements ファイル (`-p` フラグを使用) を使用する場合、変更は一切必要ありません。

**Docker**

* Docker のドキュメントは<a href="https://docs.docker.com/get-started/" target="_blank">こちらを参照 (英語)</a>

## ヒント

* サービスにデータを送る際の入力データ型にはさまざまな種類があり、サービスの呼び出しスキーマを生成するときに指定する必要があります。
* ローカル環境で DSVM とメインの Python を使用して次のコマンドを実行する場合、システムの Python に Azure ML CLI をインストールしておく必要があります (詳細は<a href="https://docs.microsoft.com/ja-jp/azure/machine-learning/desktop-workbench/deployment-setup-configuration?wt.mc_id=OH-ML-ComputerVision#using-the-cli">こちらのドキュメントを参照</a>)
    `! sudo pip install -r https://aka.ms/az-ml-o16n-cli-requirements-file`
* `az ml` コマンドで画像を作成する場合、`conda_dep.yml` やすべてのラベル ファイルなど、必要なファイルはすべて `-d` フラグを付けるようにします。  `conda_dep.yml` では、`-c` の使用は避け `-d` フラグを使用します。  また、`requirements.txt` を `pip` のインストール形式パッケージで使用する場合は、`-p` フラグを付与します。
* クラスターのデプロイの詳細は<a href="https://docs.microsoft.com/ja-jp/azure/machine-learning/desktop-workbench/deployment-setup-configuration?wt.mc_id=OH-ML-ComputerVision#environment-setup">こちらのドキュメントを参照</a>。
    
