import pandas as pd
from process import DataProcessor

def main():
    print("Starting Cloud Data Processing Service (Integrated Mode)")

    # بنجهز معالج البيانات (DataProcessor)
    processor = DataProcessor()

    # بننزل الداتاسيت من كاجل (Kaggle) - هيتحفظ أوتوماتيك في مجلد الإدخال (Input)
    processor.download_dataset()
    df = processor.load_data()

    # ------ بنحلل البيانات وبنحسب الإحصائيات الأساسية ------
    print("\n Calculating Statistics...")
    # بنستخدم نفس الدالة الموجودة في الواجهة (Gradio) عشان نحسب الإحصائيات الوصفية
    stats_data = processor.compute_descriptive_stats(df, processor.input_path)
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(stats_data.to_string(index=False))

    # ------ بنشغل مهام التعلم الآلي (Machine Learning) ------
    print("\n Running ML Jobs...")
    print("\n=== MACHINE LEARNING RESULTS ===")
    
    # بنحضر البيانات النظيفة اللي رح نستخدمها في التدريب
    clean_df = df.dropna(subset=["rating_num", "reviews_num"])
    
    # الوظيفة الأولى: الانحدار الخطي (Linear Regression)
    print("\n1) Linear Regression...")
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression
    
    reg_assembler = VectorAssembler(inputCols=["reviews_num"], outputCol="features")
    reg_df = reg_assembler.transform(clean_df)
    lr = LinearRegression(featuresCol="features", labelCol="rating_num", regParam=0.01)
    lr_model = lr.fit(reg_df)
    regression_res = f"Coeff: {lr_model.coefficients[0]:.4f} | Intercept: {lr_model.intercept:.4f}"
    print(f"   {regression_res}")
    
    # الوظيفة الثانية: التجميع (KMeans)
    print("\n2) KMeans Clustering...")
    from pyspark.ml.clustering import KMeans
    
    km_assembler = VectorAssembler(inputCols=["rating_num", "reviews_num"], outputCol="features")
    km_df = km_assembler.transform(clean_df)
    kmeans = KMeans(k=3, seed=1)
    km_model = kmeans.fit(km_df)
    centers = km_model.clusterCenters()
    
    clusters_df = pd.DataFrame({
        "Cluster": [0, 1, 2],
        "Avg Rating (Center)": [round(float(c[0]), 2) for c in centers],
        "Avg Reviews (Center)": [round(float(c[1]), 1) for c in centers]
    })
    print(clusters_df.to_string(index=False))
    
    # الوظيفة الثالثة: البحث عن الأنماط (FPGrowth)
    print("\n3) FPGrowth Pattern Mining...")
    from pyspark.ml.fpm import FPGrowth
    from pyspark.sql.functions import col, split, array_distinct
    from pyspark.sql import functions as F
    
    # بنستخدم عمود المطورين (developer) بدل عمود السعر عشان أكثر فايدة
    df_fp = df.withColumn("items", array_distinct(split(col("developer"), " "))).filter(
        F.col("developer").isNotNull() & (F.length(F.col("developer")) > 0)
    )
    
    if df_fp.count() > 0:
        fp = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.5)
        fp_model = fp.fit(df_fp)
        top_patterns = fp_model.freqItemsets.orderBy(F.col("freq").desc()).limit(3).toPandas()
        
        if not top_patterns.empty:
            top_item = top_patterns.iloc[0]
            fp_res = f"Top Pattern: {' '.join(top_item['items'][:3])} (Freq: {top_item['freq']})"
        else:
            fp_res = "No significant patterns found"
    else:
        fp_res = "Insufficient data for pattern mining"
    print(f"   {fp_res}")
    
    # الوظيفة الرابعة: الانحدار المتساوي (Isotonic Regression) - حساب الخطأ (RMSE)
    print("\n4) Isotonic Regression...")
    rmse_val = processor.isotonic_regression_job(df)
    print(f"   RMSE: {rmse_val:.4f}")

    # ------ بنفحص الأداء والقابلية للتوسع (Scalability Test) ------
    print("\n Running Performance Test...")
    cores = [1, 2, 4, 8]
    # بنكرر الاختبار 3 مرات (Repeats) عشان نضمن سرعة التنفيذ على السحابة (Cloud)
    perf_results = processor.performance_test(cores, repeats=3)

    # بنجهز جدول الأداء عشان التقرير
    print("\n=== PERFORMANCE & SCALABILITY ===")
    perf_rows = []
    for c, t, s, e, times in perf_results:
        perf_rows.append({
            "Cores": c, 
            "Time (s)": f"{t:.3f}", 
            "Speedup": f"{s:.2f}x", 
            "Efficiency": f"{e*100:.1f}%"
        })
    perf_df = pd.DataFrame(perf_rows)
    print(perf_df.to_string(index=False))

    # ------ بنحفظ كل النتائج في ملف نصي واحد في مجلد الإخراج (Output) ------
    print("\n Saving all results to Output folder...")
    save_msg = processor.save_all_results_to_file(
        stats_df=stats_data,
        reg_res=regression_res,
        kmeans_df=clusters_df,
        fp_res=fp_res,
        iso_val=rmse_val,
        perf_df=perf_df
    )

    print(f"\n{save_msg}")
    
    # ------ بنرفع النتائج على قاعدة البيانات (Firebase) ------
    print("\n Uploading results to Firebase...")
    firebase_status = processor.save_to_firebase(
        stats_data, regression_res, clusters_df, fp_res, rmse_val, perf_df
    )
    print(f"{firebase_status}")
    
    print("\n Job completed successfully!")

if __name__ == "__main__":
    main()
