# ⛰ "Ölçekli Eğitim" Ünitesi 🗻

Bu ünitede, WorkinTech'teki Veri Bilimi ekibi tarafından sağlanan notebook'u nasıl paketleyeceğinizi ve tam veri seti üzerinde yerel olarak eğitilebilmesi için nasıl ölçeklendireceğinizi öğreneceksiniz.

Bu ünite aşağıdaki 5 challenge'dan oluşur ve hepsi bu tek `README` dosyasında gruplandırılmıştır.

Sadece kılavuzu takip edin ve ilerlemelerinizi izleyebilmemiz için her ana bölümden sonra `git push` yapın!

# 1️⃣ Yerel Kurulum

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

Projenin baş ML Mühendisi olarak, ilk rolünüz yerel bir çalışma ortamı (pyenv ile) ve yalnızca kod tabanınızın iskeletini içeren bir python paketi kurmaktır.

💡 Notebook'ları paketlemek temel bir ML Mühendisi becerisidir. Bu şunları sağlar:
- diğer kullanıcıların kod üzerinde işbirliği yapabilmesi
- kodu yerel olarak veya uzak bir makinede klonlayabilmeniz, örneğin `taxifare` modelini daha güçlü bir makinede eğitmek için
- kodu bir **API** olarak veya bir **web sitesi** aracılığıyla sunmak üzere üretime koyabilmeniz
- kodu manuel olarak çalıştırılabilir veya otomasyon iş akışına entegre edilebilir hale getirebilmeniz

### 1.1) [🐍 taxifare-env] adında yeni bir pyenv oluşturun

WorkinTech'in ML yığını Python 3.10.6 üzerinde çalışıyor, WorkinTech'e iyi hizmet veren kararlı bir sürüm: *"Bozuk değilse, tamir etme."* O halde aynı Python sürümünü kullanalım.

🐍 Python 3.10.6'yı kurun

```bash
pyenv install 3.10.6
```

🐍 Virtual ortamı oluşturun

```bash
cd ~/code/{{local_path_to("07-ML-Ops/01-Train-at-scale/01-Train-at-scale")}}
```

```bash
pyenv virtualenv 3.10.6 taxifare-env
pyenv local taxifare-env
pip install --upgrade pip
code .
```

Daha sonra, hem işletim sisteminizin Terminali hem de VS Code'un entegre Terminalinin `[🐍 taxifare-env]` gösterdiğinden emin olun.
VS Code'da herhangi bir `.py` dosyası açın ve aşağıda gösterildiği gibi sağ alttaki pyenv bölümüne tıklayarak `taxifare-env`'in aktif olduğunu kontrol edin:

<a href="/pyenv-setup.png" target="_blank">
    <img src='/pyenv-setup.png' width=400>
</a>

### 1.2) taxifare paket yapısına aşina olun

❗️Sizin için hazırladığımız şablonun yapısını anlamak için 10 dakika ayırın (detaya girmeyin); giriş noktası `taxifare.interface.main_local`'dir: hızlıca takip edin.

```bash
. # Challenge klasör kökü
├── Makefile          # 🚪 Komut "başlatıcınız". Yaygın olarak kullanın (eğitim başlatma, testler, vb...)
├── README.md         # Şu anda okuduğunuz dosya!
├── notebooks
│   └── datascientist_deliverable.ipynb   # DS ekibinden teslim edilecek dosya!
├── requirements.txt   # Yerel ortamınıza eklenecek tüm üçüncü taraf paketleri listeler
├── setup.py           # Paketiniz için `pip install`'ı etkinleştirir
├── taxifare           # Bu paketin kod mantığı
│   ├── __init__.py
│   ├── interface
│   │   ├── __init__.py
│   │   └── main_local.py  # 🚪 Tüm "rotaları" içeren ana Python giriş noktanız
│   └── ml_logic
│   |    ├── __init__.py
│   |    ├── data.py           # Veri kaydetme, yükleme ve temizleme
│   |    ├── encoders.py       # Özel encoder yardımcıları
│   |    ├── model.py          # TensorFlow modeli
│   |    ├── preprocessor.py   # Sklearn ön işleme ardışık düzenleri
│   |    ├── registry.py       # Modelleri kaydetme ve yükleme
|   ├── utils.py    # # taxifare mantığına bağımlı olmayan yararlı python fonksiyonları
|   ├── params.py   # Global proje parametreleri
|
├── tests  # `make test_...` kullanarak çalıştırılacak testler
│   ├── ...
│   └── ...
├── .gitignore
```

🐍 Paketinizi bu yeni virtual ortama kurun

```bash
cd ~/code/{{local_path_to("07-ML-Ops/01-Train-at-scale/01-Train-at-scale")}}
pip install -e .
```

`pip list | grep taxifare` çalıştırarak paketin kurulduğundan emin olun; paketin mutlak yolunu yazdırmalıdır.


### 1.3) Veri nerede?

**Ham veri Google Big Query'de**

WorkinTech'ın mühendislik ekibi, 2009'dan beri tüm taksi yolculuğu geçmişini devasa bir Big Query tablosunda `data-analytics-469406.mlops.mlops_all` saklar.

**Google Cloud Platform erişimini kontrol edin**

🧪 Her şeyin yolunda olduğunu kontrol edin
```bash
make test_gcp_setup
```

**BQ'yu iki kez sorgulamaktan kaçınmak için tüm ara verileri her zaman yerel olarak `~/.workintech/mlops/` içinde önbellekte saklarız**

💾 `data` klasörümüzü bu challenge klasörünün *dışında* saklayalım, böylece tüm ML Ops modülü boyunca diğer tüm challenge'lar tarafından erişilebilir. Zaten `git` tarafından takip edilmesini istemiyoruz!

``` bash
# Create the data folder
mkdir -p ~/workintech/mlops/data/

# Create relevant subfolders
mkdir ~/workintech/mlops/data/raw
mkdir ~/workintech/mlops/data/processed
```

💡Buradayken, tüm challenge'lar tarafından paylaşılacak `training_outputs` için de bir depolama klasörü oluşturalım

```bash
# Create the training_outputs folder
mkdir ~/workintech/mlops/training_outputs

# Create relevant subfolders
mkdir ~/workintech/mlops/training_outputs/metrics
mkdir ~/workintech/mlops/training_outputs/models
mkdir ~/workintech/mlops/training_outputs/params
```

Artık gelecek challenge'lar için verilerin, Veri Bilimi ekibinin notebook'ları ve model çıktıları ile birlikte `~/workintech/mlops/` içinde depolandığını görebilirsiniz:

``` bash
tree -a ~/workintech/mlops/

# BUNU GÖRMELİSİNİZ
├── data          # Burada şunları yapacaksınız:
│   ├── processed # Ara, işlenmiş verileri saklama
│   └── raw       # Ham veri örneklerini indirme
└── training_outputs
    ├── metrics # Eğitilmiş model metriklerini saklama
    ├── models  # Eğitilmiş model ağırlıklarını saklama (büyük olabilir!)
    └── params  # Eğitilmiş model hiperparametrelerini saklama
```

☝️ İstediğiniz zaman tüm dosyaları kaldırıp bu boş klasör yapısını korumak için şunu kullanabilirsiniz

```bash
make reset_local_files
```

</details>

# 2️⃣ Veri Bilimcisinin Çalışmasını Anlama

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

*⏱ Süre: buna 1 saat ayırın*

🖥️ `datascientist_deliverable.ipynb`'i VS Code ile açın (bu modül için Jupyter'i unutun) ve tüm hücreleri dikkatli bir şekilde çalıştırın, onları anlarken. Sizinle DS ekibi arasındaki bu devir teslim, onlarla etkileşime geçmek (yani arkadaşınız veya bir TA) için mükemmel zaman.

❗️`taxifare-env`'i bir `ipykernel` venv olarak kullandığınızdan emin olun

<a href="/pyenv-notebook.png" target="_blank">
    <img src='/pyenv-notebook.png' width=400>
</a>

</details>


# 3️⃣ Kodunuzu Paketleyin

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

🎯 Hedefiniz aşağıda görüldüğü gibi `taxifare.interface.main_local` modülünü çalıştırabilmektir

```bash
# -> model
python -m taxifare.interface.main_local
```

🖥️ Bunu yapmak için, lütfen aşağıdaki dosyalarda `# YOUR CODE HERE` ile işaretlenmiş eksik bölümleri kodlayın; Notebook'u oldukça yakından takip etmelidir!

```bash
├── taxifare
│   ├── __init__.py
│   ├── interface
│   │   ├── __init__.py
│   │   └── main_local.py   # 🔵 🚪 Entry point: code both `preprocess_and_train()` and `pred()`
│   └── ml_logic
│       ├── __init__.py
│       ├── data.py          # 🔵 your code here
│       ├── encoders.py      # 🔵 your code here
│       ├── model.py         # 🔵 your code here
│       ├── preprocessor.py  # 🔵 your code here
│       ├── registry.py  # ✅ `save_model` and `load_model` are already coded for you
|   ├── params.py # 🔵 You need to fill your GCP_PROJECT
│   ├── utils.py
```

**🧪 Kodunuzu test edin**

Şu anki taxifare-env ortamında paketin doğru kurulduğundan emin olun, eğer değilse

```bash
pip list | grep taxifare
```

Daha sonra, paketinizin `python -m taxifare.interface.main_local` ile düzgün çalıştığından emin olun.
- Çalışana kadar hata ayıklama yapın!
- Aşağıdaki veri seti boyutlarını kullanın

```python
# taxifare/ml_logic/params.py
DATA_SIZE = '1k'   # To iterate faster in debug mode 🐞
DATA_SIZE = '200k' # Should work at least once
# DATA_SIZE = 'all' 🚨 DON'T TRY YET, it's too big and will cost money!
```

Daha sonra, yalnızca `make test_preprocess_and_train` ile testleri geçmeye çalışın!

✅ Tüm testler yeşil olduğunda, sonuçlarınızı kitt'te `make test_kitt` ile takip edin

</details>

# 4️⃣ Ölçeklenebilirliği Araştırın

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

*⏱ Süre: buna en fazla 20 dakika ayırın*

Artık paketi küçük bir veri seti için çalıştırmayı başardığınıza göre, gerçek veri setiyle nasıl baş a çıkacağını görme zamanı!

👉 Ciddiye almaya başlamak için `ml_logic.params.DATA_SIZE`'i `all` olarak değiştirin!

🕵️ Kodunuzun hangi bölümünün **en çok zaman** aldığını ve **en çok bellek** kullandığını `taxifare.utils.simple_time_and_memory_tracker`'ı seçtiğiniz metodlara dekore etmek için kullanarak araştırın.

```python
# taxifare.ml_logic.data.py
from taxifare.utils import simple_time_and_memory_tracker

@simple_time_and_memory_tracker
def clean_data() -> pd.DataFrame:
    ...
```

🕵️ Arkadaşınızla aşağıdaki soruları cevaplamaya çalışın:
- Kodunuzun hangi bölümü temel dar boğazları barındırır?
- Hangi tür dar boğazlar en endişe verici? (zaman? bellek?)
- Size 50M satır verseydi ölçekleneceğini düşünüyor musunuz? 500M? Bu arada, [gerçek NYC veri seti](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) daha da büyük ve yaklaşık 156GB ağırlığında!
- Potansiyel çözümler hakkında düşünebiliyor musunuz? Fikirlerinizi yazın, ama henüz uygulama yapmayın!
</details>


# 5️⃣ Arttırımlı İşleme

<details>
  <summary markdown='span'><strong>❓Talimatlar (beni genişlet)</strong></summary>

🎯 Hedefiniz kod tabanınızı sınırsız miktarda satır üzerinde model eğitebilecek şekilde geliştirmektir, **RAM sınırlarına ulaşmadan**, tek bir bilgisayarda.

## 5.1) Tartışma

**Ne öğrendik?**

Bellek ve zaman kısıtlarımız var:
- `(55M, 8)` boyutlu ham veri bellek içinde DataFrame olarak yüklendirilir ve yaklaşık 10GB RAM kullanır, bu da çoğu bilgisayar için çok fazladır.
- `(55M, 65)` boyutlu ön işlemli DataFrame daha da büyüktür.
- `ml_logic.encoders.compute_geohash` metodu işleme çok uzun zaman alır 🤯

Bir çözüm yeterli RAM'i olan bir *bulut Sanal Makinesi (VM)* için para ödeyerek onu orada işlemektir (böyle bir problemle baş etmenin genellikle en basit yolu budur).

**Önerilen çözüm: arttırımlı ön işleme 🔪 parça parça 🔪**

<img src="/process_by_chunk.png" width=500>

💡 Ön işlemcimiz *durumsuz* olduğundan, kolayca şunları yapabiliriz:
- Herhangi bir _sütun bazlı istatistik_ hesaplamasından kaçınıp yalnızca _satır bazında ön işleme_ gerçekleştirin
- _ön işleme_'yi _eğitim_'den ayırın ve herhangi bir ara sonuç diskte saklayın!

🙏 Bu nedenle, sınırlı boyutta parçalar (Ör. 100.000 satır) ile ön işlemeyi *parça parça* yapalım, her parça belleğe güzelçe sığacak:

1. `data_processed_chunk_01`'i sabit diskte saklarayız.
2. Sonra `data_processed_chunk_02`'yi ilkine ekleriz.
3. vb...
4. `~/workintech/mlops/data/processed/processed_all.csv` adresinde devasa bir CSV depolanana kadar

5. Bölüm 6️⃣'te, modelimizi de parça parça `train()` ederiz, her parçada yükleme ve eğitimi tekrarlı olarak yaparak (sonraki bölümde daha fazlası)

## 5.2) Sıranız: `def preprocess()` kodlayın

👶 **Önce, hata ayıklama amaçları için daha küçük veri seti boyutlarını geri getirelim**

```python
# params.py
DATA_SIZE = '1k'
CHUNK_SIZE = 200
```

**Daha sonra, `ml_logic.interface.main_local` modülünüzdeki `def preprocess()` ile aşağıda verilen yeni rotayı kodlayın; başlamak için aşağıdaki kodu kopyalayıp yapıştırın**

[//]: # (  🚨 Code below is NOT the single source of truth. Original is in data-solutions repo 🚨 )

<br>

<details>
  <summary markdown='span'>👇 Kopyalanacak kod 👇</summary>

```python
def preprocess(min_date: str = '2009-01-01', max_date: str = '2015-01-01') -> None:
    """
    Query and preprocess the raw dataset iteratively (by chunks).
    Then store the newly processed (and raw) data on local hard-drive for later re-use.

    - If raw data already exists on local disk:
        - use `pd.read_csv(..., chunksize=CHUNK_SIZE)`

    - If raw data does not yet exists:
        - use `bigquery.Client().query().result().to_dataframe_iterable()`

    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess by batch" + Style.RESET_ALL)

    from taxifare.ml_logic.data import clean_data
    from taxifare.ml_logic.preprocessor import preprocess_features

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM `{GCP_PROJECT_WORKINTECH}`.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """
    # Retrieve `query` data as dataframe iterable
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")

    data_query_cache_exists = data_query_cache_path.is_file()
    if data_query_cache_exists:
        print("Get a dataframe iterable from local CSV...")
        chunks = None
        # YOUR CODE HERE

    else:
        print("Get a dataframe iterable from Querying Big Query server...")
        chunks = None
        # 🎯 Hints: `bigquery.Client(...).query(...).result(page_size=...).to_dataframe_iterable()`
        # YOUR CODE HERE

    for chunk_id, chunk in enumerate(chunks):
        print(f"processing chunk {chunk_id}...")

        # Clean chunk
        # YOUR CODE HERE

        # Create chunk_processed
        # 🎯 Hints: Create (`X_chunk`, `y_chunk`), process only `X_processed_chunk`, then concatenate (X_processed_chunk, y_chunk)
        # YOUR CODE HERE

        # Save and append the processed chunk to a local CSV at "data_processed_path"
        # 🎯 Hints: df.to_csv(mode=...)
        # 🎯 Hints: We want a CSV without index nor headers (they'd be meaningless)
        # YOUR CODE HERE

        # Save and append the raw chunk if not `data_query_cache_exists`
        # YOUR CODE HERE
    print(f"✅ data query saved as {data_query_cache_path}")
    print("✅ preprocess() done")


```

</details>

<br>

**❓Aşağıdaki ön işlemli veri setlerini oluşturup saklamaya çalışın**

- `preprocess()` çalıştırarak `data/processed/train_processed_1k.csv`

<br>

**🧪 Kodunuzu test edin**

Kodunuzu `make test_preprocess_by_chunk` ile test edin.

✅ Tüm testler yeşil olduğunda, sonuçlarınızı kitt'te `make test_kitt` ile takip edin

<br>

**❓Son olarak, gerçek ön işlemli veri setlerini oluşturup saklayın**

Kullanarak:
```python
# params.py
DATA_SIZE = 'all'
CHUNK_SIZE = 100000
```

🎉 Birkaç saatlik hesaplama ile 55 Milyon satırı da kolayca işleyebilirdik, ama bugün yapmayaalım 😅

</details>

# 6️⃣ Arttırımlı Öğrenme

<details>
  <summary markdown='span'><strong>❓Talimatlar (beni genişlet)</strong></summary>

<br>

🎯 Hedef: modelimizi tam `.../processed/processed_all.csv` üzerinde eğitmek

## 6.1) Tartışma

Teorik olarak, `(xxMilyonlar, 65)` boyutundaki böyle büyük bir veri setini aynı anda RAM'e yükleyemeyiz, ama parçalar halinde yükleyebiliriz.

**Bir modeli parçalar halinde nasıl eğitiriz?**

Buna **arttırımlı öğrenme** veya **partial_fit** denir
- Rastgele ağırlıklarla bir model başlatırız ${\theta_0}$
- İlk `data_processed_chunk`'i belleğe yükleriz (diyelim ki 100.000 satır)
- Modelimizi ilk parça üzerinde eğitiriz ve ağırlıklarını buna göre güncelleriz ${\theta_0} \rightarrow {\theta_1}$
- İkinci `data_processed_chunk`'i belleğe yükleriz
- Modelimizi ikinci parça üzerinde *yeniden eğitiriz*, bu sefer önceden hesaplanmış ağırlıkları güncelleriz ${\theta_1} \rightarrow {\theta_2}$!
- Veri setinin sonuna kadar tekrarlarız

❗️Tüm Makine Öğrenmesi modelleri arttırımlı öğrenmeyi desteklemez; yalnızca Gradyan İnişi gibi *iteratif güncelleme metotları*na dayalı *parametrik* modeller $f_{\theta}$ bunu destekler
- **scikit-learn**'de, `model.partial_fit()` yalnızca SGDRegressor/Classifier ve birkaç diğeri için mevcuttur ([bunu dikkatli oku 📚](https://scikit-learn.org/0.15/modules/scaling_strategies.html#incremental-learning)).
- **TensorFlow** ve diğer Derin Öğrenme çerçevelerinde, eğitim her zaman iteratiftir ve arttırımlı öğrenme varsayılan davranıştır! Yalnızca iki parça arasında `model.initialize()` çağırmaktan kaçınmanız gerekir!

❗️Derin Öğrenmedeki `chunk_size`'ı `batch_size` ile karıştırmayın

👉 Her (büyük) parça için, modeliniz verileri birkaç epoch boyunca birden çok (küçük) batch'te okuyacaktır

<img src='/train_by_chunk.png'>

👍 **Avantajlar:** bu evrensel yaklaşım çerçeve-bağımsızdır; `scikit-learn`, XGBoost, TensorFlow vb. ile kullanabilirsiniz.

👎 **Dezavantajlar:** model *en son* parçaya *ilk* olanlardan daha iyi uyacak şekilde önyargılı olacaktır. Bizim durumumuzda, eğitim veri setimiz karıştırılmış olduğu için bu bir sorun değil, ancak modeli üretimde olduktan sonra yeni verilerle kısmi uydurmak yaparken bunu akılda tutmak önemlidir.

<br>

<details>
  <summary markdown='span'><strong>🤔 TensorFlow ile gerçekten parçalara ihtiyacımız var mı?</strong></summary>

Elbette, TensorFlow veri setleri sayesinde, aşağıda görüldüğü gibi batch-by-batch veri seti yüklemesini kullanabileceğiniz için her zaman "parçalara" ihtiyacınız olmayacaktır

```python
import tensorflow as tf

ds = tf.data.experimental.make_csv_dataset(data_processed_all.csv, batch_size=256)
model.fit(ds)
```

Bunu Recap'te göreceğiz. Yine de, bu challenge'da size herhangi bir çerçeveye uygulanan evrensel parçalarda arttırımlı uyum metotunu öğretmek istiyoruz ve bu model üretime konulduktan sonra modelinizi yeni verilerle *kısmen yeniden eğitmek* için yararlı olacaktır.
</details>

<br>

## 6.2) Sıranız - `def train()` kodlayın

**`ml_logic.interface.main_local` modülünüzdeki `def train()` ile aşağıda verilen yeni rotayı kodlamaya çalışın; başlamak için aşağıdaki kodu kopyalayıp yapıştırın**

Yine, çok küçük bir veri seti boyutu ile başlayın, daha sonra modelinizi 500k satır üzerinde eğitin.

[//]: # (  🚨 Code below is not the single source of truth 🚨 )

<details>
  <summary markdown='span'><strong>👇 Kopyalanacak kod 👇</strong></summary>

```python
def train(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    Incremental train on the (already preprocessed) dataset locally stored.
    - Loading data chunk-by-chunk
    - Updating the weight of the model for each chunk
    - Saving validation metrics at each chunks, and final model weights on local disk
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case:train by batch" + Style.RESET_ALL)
    from taxifare.ml_logic.registry import save_model, save_results
    from taxifare.ml_logic.model import (compile_model, initialize_model, train_model)

    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    model = None
    metrics_val_list = []  # store each val_mae of each chunk

    # Iterate in chunks and partial fit on each chunk
    chunks = pd.read_csv(data_processed_path,
                         chunksize=CHUNK_SIZE,
                         header=None,
                         dtype=DTYPES_PROCESSED)

    for chunk_id, chunk in enumerate(chunks):
        print(f"training on preprocessed chunk n°{chunk_id}")
        # You can adjust training params for each chunk if you want!
        learning_rate = 0.0005
        batch_size = 256
        patience=2
        split_ratio = 0.1 # Higher train/val split ratio when chunks are small! Feel free to adjust.

        # Create (X_train_chunk, y_train_chunk, X_val_chunk, y_val_chunk)
        train_length = int(len(chunk)*(1-split_ratio))
        chunk_train = chunk.iloc[:train_length, :].sample(frac=1).to_numpy()
        chunk_val = chunk.iloc[train_length:, :].sample(frac=1).to_numpy()

        X_train_chunk = chunk_train[:, :-1]
        y_train_chunk = chunk_train[:, -1]
        X_val_chunk = chunk_val[:, :-1]
        y_val_chunk = chunk_val[:, -1]

        # Train a model *incrementally*, and store the val MAE of each chunk in `metrics_val_list`
        # YOUR CODE HERE

    # Return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    # Save model and training params
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        incremental=True,
        chunk_size=CHUNK_SIZE
    )

    print(f"✅ Trained with MAE: {round(val_mae, 2)}")

     # Save results & model
    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ train() done")

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    from taxifare.ml_logic.registry import load_model
    from taxifare.ml_logic.preprocessor import preprocess_features

    if X_pred is None:
       X_pred = pd.DataFrame(dict(
           pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
           pickup_longitude=[-73.950655],
           pickup_latitude=[40.783282],
           dropoff_longitude=[-73.984365],
           dropoff_latitude=[40.769802],
           passenger_count=[1],
       ))

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print(f"✅ pred() done")
    return y_pred

```

</details>

**🧪 Kodunuzu test edin**

`make test_train_by_chunk` ile kontrol edin

🏁 🏁 🏁 🏁 Tebrikler! 🏁 🏁 🏁 🏁


</details>
