import os
import time
import firebase_admin
from firebase_admin import credentials, db
import kagglehub
from kagglehub import KaggleDatasetAdapter

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, split, array_distinct, expr
)

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression, IsotonicRegression
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.evaluation import RegressionEvaluator

def trimmed_median(values, trim=0.2):
    """بنحسب الوسيط القوي (Trimmed Median) عشان نفحص الأداء ونتأكد من دقة النتائج"""
    vals = sorted(values)
    n = len(vals)
    if n == 0: return None
    k = int(n * trim)
    trimmed = vals[k:n-k] if n - 2*k >= 1 else vals
    m = len(trimmed)
    mid = m // 2
    return trimmed[mid] if m % 2 == 1 else (trimmed[mid-1] + trimmed[mid]) / 2.0

class DataProcessor:
    # دالة البداية عشان نجهز المسارات ونشغل السبارك (Spark)
    def __init__(self):
        # مجلد الإدخال (Input) عشان الداتا ومجلد الإخراج (Output) عشان النتائج
        # بنجيب المسار من متغير البيئة أو نستخدم المسار النسبي
        project_root = os.getenv('PROJECT_BASE_PATH', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.base_path = os.path.join(project_root, 'data_and_result')
        self.input_path = os.path.join(self.base_path, 'input', 'data.csv')
        self.output_path = os.path.join(self.base_path, 'output')
        self.report_path = os.path.join(self.output_path, 'Final_Analysis_Report.txt')

        os.makedirs(os.path.join(self.base_path, 'input'), exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

        # بننشئ جلسة السبارك (Spark Session) للمعالجة الموزعة
        self.spark = SparkSession.builder \
            .appName("ShopifyAppStoreAnalytics") \
            .config("spark.sql.shuffle.partitions", "64") \
            .getOrCreate()
        
        # بنجهز الاتصال بقاعدة البيانات (Firebase) باستخدام ملف المفتاح
        try:
            cred_path = os.getenv('FIREBASE_KEY_PATH', os.path.join(project_root, 'firebase_key.json'))
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                # بنحط رابط قاعدة البيانات الفايربيس (Firebase Realtime Database)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': 'https://cloudeproject-fabea-default-rtdb.firebaseio.com/' 
                })
            self.db_ref = db.reference('/')
            print("Connected to Firebase Successfully")
        except Exception as e:
            print(f"Firebase Init Warning: {e}")

    def save_to_firebase(self, stats_df, reg_res, kmeans_df, fp_res, iso_val, perf_df):
        """رفع النتائج بما فيها نواتج التعلم الآلي (Machine Learning) لقاعدة البيانات"""
        try:
            # بنحول كل الجداول لقواميس (Dictionary) عشان نقدر نرفعها
            payload = {
                "timestamp": time.ctime(),
                "processed_file_type": self.input_path.split('.')[-1],
                "summary": stats_df.to_dict(orient='records'),
                
                # نواتج التعلم الآلي (Machine Learning) هنا
                "machine_learning": {
                    "linear_regression": str(reg_res), # وصف نموذج الانحدار الخطي
                    "kmeans_centers": kmeans_df.to_dict(orient='records'), # مراكز مجموعات الكي-مينز (KMeans)
                    "fpgrowth": str(fp_res), # أنماط الإف-بي-جروث (FPGrowth)
                    "isotonic_rmse": float(iso_val) # قيمة الخطأ (RMSE) للانحدار المتساوي
                },
                
                "performance": perf_df.to_dict(orient='records')
            }
            
            # بنرفع البيانات على الفايربيس (Firebase) في قسم التحليلات
            self.db_ref.child('analytics').set(payload)
            return "✅ Machine Learning Results Stored!"
        except Exception as e:
            return f"❌ Firebase ML Storage Error: {str(e)}"
    def download_dataset(self):
        """بننزل الداتاسيت من موقع كاجل (Kaggle) ونحفظه في مجلد الإدخال (Input)"""
        print("Downloading dataset...")
        df_pandas = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "usernam3/shopify-app-store",
            "apps.csv"
        )
        df_pandas.to_csv(self.input_path, index=False)
        print(f"Dataset saved to {self.input_path}")

    def stage_uploaded_file(self, uploaded_path):
        """نسخ الملف المرفوع لمجلد الإدخال (Input) في الدرايف (Drive) وإرجاع المسار الجديد"""
        import shutil
        filename = os.path.basename(uploaded_path)
        destination = os.path.join(self.base_path, 'input', filename)
        shutil.copy2(uploaded_path, destination)
        print(f"File staged to {destination}")
        return destination

    # دالة عشان نقرا البيانات من الملف ونحولها لأعمدة رقمية
    def load_data(self):
        """قراءة البيانات سواء كانت (CSV) أو (JSON) أو نص (TXT) وتحويلها لجدول سبارك (Spark DataFrame)"""
        file_ext = self.input_path.lower()

        if file_ext.endswith(".csv"):
            # بنقرا ملفات الـ سي-إس-في (CSV)
            df = self.spark.read.csv(self.input_path, header=True, inferSchema=True)
        
        elif file_ext.endswith(".json"):
            # بنقرا ملفات الجيسون (JSON) سواء سطر واحد أو ملف كامل
            df = self.spark.read.json(self.input_path)
        
        elif file_ext.endswith(".txt"):
            # بنقرا ملفات النص (TXT) المفصولة بتاب (Tab) أو مسافات
            # أو كملف سي-إس-في (CSV) مع تحديد الفاصل (Delimiter)
            df = self.spark.read.option("delimiter", "\t").csv(self.input_path, header=True, inferSchema=True)
        
        else:
            raise ValueError("Unsupported file format! Please use CSV, JSON, or TXT.")

        # بنحول الأعمدة لأرقام عشان نقدر نشغل نماذج التعلم الآلي (ML)
        df = df.withColumn("rating_num", expr("try_cast(rating as double)")) \
               .withColumn("reviews_num", expr("try_cast(reviews_count as double)"))
        return df

    def compute_descriptive_stats(self, df, file_path=None):
        """
        ترجع جدول ملخص (Summary) مبسط يحتوي على 6 إحصائيات وصفية
        """
        import pandas as pd
        import os

        # بنحسب إجمالي عدد الصفوف (Rows)
        total_rows = df.count()
        
        # بنحسب عدد الأعمدة (Columns)
        n_cols = len(df.columns)

        # بنحدد نوع البيانات (رقمية أو نصية)
        original_cols = [c for c in df.columns if c not in ["rating_num", "reviews_num"]]
        dtypes = [(c, t) for c, t in df.dtypes if c in original_cols]
        numeric_types = {"int", "bigint", "double", "float", "decimal", "smallint", "tinyint", "long", "short"}
        has_numeric = any(t in numeric_types for _, t in dtypes)
        has_text = any(t not in numeric_types for _, t in dtypes)
        
        if has_numeric and has_text:
            data_types = "Numeric & Text"
        elif has_numeric:
            data_types = "Numeric"
        else:
            data_types = "Text"

        # بنحسب الحد الأدنى والأقصى للتقييمات (Min/Max Rating) من البيانات النظيفة
        clean_df = df.filter(
            F.col("rating_num").isNotNull() & 
            (F.col("rating_num") > 0)
        )
        
        if clean_df.count() > 0:
            stats = clean_df.select(
                F.max("rating_num").alias("max_rating"),
                F.min("rating_num").alias("min_rating")
            ).first()
            
            max_rating = float(stats["max_rating"]) if stats["max_rating"] else 5.0
            min_rating = float(stats["min_rating"]) if stats["min_rating"] else 0.0
        else:
            max_rating = 5.0
            min_rating = 0.0

        # بنحسب حجم الملف بالميجابايت (MB)
        file_size_mb = None
        if file_path and os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        # بنبني الجدول النهائي اللي فيه الـ 6 إحصائيات
        rows = []
        if file_size_mb is not None:
            rows.append(("Dataset Size (MB)", f"{file_size_mb:.2f}"))

        rows += [
            ("Data Types", data_types),
            ("Min Rating", f"{min_rating:.1f}"),
            ("Max Rating", f"{max_rating:.1f}"),
            ("Total Rows", f"{total_rows:,}"),
            ("Total Columns", f"{n_cols}"),
        ]

        return pd.DataFrame(rows, columns=["Metric", "Value"])

    def save_all_results_to_file(self, stats_df, reg_res, kmeans_df, fp_res, iso_val, perf_df):
        """دالة بتجمع كل مخرجات المشروع في ملف نصي واحد في مجلد الإخراج (Output)"""
        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                f.write("==================================================\n")
                f.write("PROJECT ANALYSIS REPORT (CLOUD COMPUTING)\n")
                f.write(f"Report Date: {time.ctime()}\n")
                f.write("==================================================\n\n")

                f.write("[1] DATA STATISTICS\n")
                f.write(stats_df.to_string(index=False) + "\n\n")

                f.write("[2] MACHINE LEARNING RESULTS\n")
                f.write(f"- Linear Regression Model: {reg_res}\n\n")
                f.write(f"- KMeans Clusters (Centers):\n{kmeans_df.to_string(index=False)}\n\n")
                f.write(f"- FPGrowth Patterns: {fp_res}\n\n")
                f.write(f"- Isotonic Regression RMSE: {iso_val:.4f}\n\n")

                f.write("[3] PERFORMANCE BENCHMARKS\n")
                f.write(perf_df.to_string(index=False) + "\n\n")

                f.write("==================================================\n")
                f.write("REPORT COMPLETED\n")
            return f"Report saved at: {self.report_path}"
        except Exception as e:
            return f"Error saving file: {str(e)}"

    # دالة عشان نعمل انحدار خطي (Linear Regression) ونشوف العلاقة بين المراجعات والتقييمات
    def regression_job(self, df):
        df_reg = df.dropna(subset=["rating_num", "reviews_num"])
        assembler = VectorAssembler(inputCols=["reviews_num"], outputCol="features")
        df_reg = assembler.transform(df_reg)
        lr = LinearRegression(featuresCol="features", labelCol="rating_num", regParam=0.01)
        model = lr.fit(df_reg)
        return f"Coeff: {model.coefficients[0]:.4f} | Intercept: {model.intercept:.4f}"

    # دالة عشان نقسم التطبيقات لـ 3 مجموعات باستخدام خوارزمية الكي-مينز (KMeans) حسب التقييم والمراجعات
    def kmeans_job(self, df):
        df_km = df.dropna(subset=["rating_num", "reviews_num"])
        assembler = VectorAssembler(inputCols=["rating_num", "reviews_num"], outputCol="features")
        df_km = assembler.transform(df_km)
        model = KMeans(k=3, seed=1).fit(df_km)
        import pandas as pd
        centers = []
        for i, c in enumerate(model.clusterCenters()):
            centers.append({"Cluster": i, "Avg Rating": round(float(c[0]), 2), "Avg Reviews": round(float(c[1]), 1)})
        return pd.DataFrame(centers)

    # دالة عشان نلاقي الأنماط المتكررة باستخدام خوارزمية الإف-بي-جروث (FPGrowth) في أسعار التطبيقات
    def fpgrowth_job(self, df):
        df_fp = df.withColumn("items", array_distinct(split(col("pricing_hint"), " "))).dropna(subset=["items"])
        fp = FPGrowth(itemsCol="items", minSupport=0.2)
        model = fp.fit(df_fp)
        top = model.freqItemsets.limit(1).toPandas()
        if not top.empty:
            return f"Pattern: {top['items'].iloc[0]} (Freq: {top['freq'].iloc[0]})"
        return "No patterns found"

    # دالة عشان نعمل انحدار متساوي (Isotonic Regression) ونحسب دقة التوقعات
    def isotonic_regression_job(self, df):
        df_iso = df.dropna(subset=["rating_num", "reviews_num"])
        assembler = VectorAssembler(inputCols=["reviews_num"], outputCol="features")
        df_ml = assembler.transform(df_iso)
        ir = IsotonicRegression(labelCol="rating_num", featuresCol="features")
        model = ir.fit(df_ml)
        predictions = model.transform(df_ml)
        evaluator = RegressionEvaluator(labelCol="rating_num", predictionCol="prediction", metricName="rmse")
        return evaluator.evaluate(predictions)

    # دالة عشان نفحص أداء المعالج بعدد كورات (Cores) مختلف ونقيس السرعة
    def performance_test(self, cores_list, repeats=3):
        df = self.load_data().dropna(subset=["rating_num", "reviews_num"])
        assembler = VectorAssembler(inputCols=["reviews_num"], outputCol="features")
        train = assembler.transform(df).select("features", col("rating_num").alias("label")).cache()
        train.count()
        
        lr = LinearRegression(featuresCol="features", labelCol="label", regParam=0.01, maxIter=5)
        base_time = None
        results = []
        for cores in cores_list:
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                lr.fit(train)
                times.append(time.perf_counter() - t0)
            
            exec_time = trimmed_median(times)
            if base_time is None: base_time = exec_time
            speedup = base_time / exec_time
            results.append((cores, exec_time, speedup, speedup/cores, times))
        train.unpersist()
        return results
