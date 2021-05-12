df = spark.createDataFrame([ (1, 'A',12560,45),
                             (1, 'B',42560,90),
                             (1, 'C',31285,120),
                             (1, 'D',10345,150)
                           ], ["userID", "Name","Revenue","No_of_Days"])

print("Before Scaling :")
df.show(5)

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

# UDF for converting column type from vector to double type
unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

# Iterating over columns to be scaled
for i in ["Revenue","No_of_Days"]:
    # VectorAssembler Transformation - Converting column to vector type
    assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")

    # MinMaxScaler Transformation
    scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")

    # Pipeline of VectorAssembler and MinMaxScaler
    pipeline = Pipeline(stages=[assembler, scaler])

    # Fitting pipeline on dataframe
    fittedpipeline = pipeline.fit(df)
    df = fittedpipeline.transform(df).withColumn(i+"_Scaled", unlist(i+"_Scaled")).drop(i+"_Vect")

    print("Inversing column: y_scaled")
    Xmax = fittedpipeline.stages[-1].originalMax[0]
    Xmin = fittedpipeline.stages[-1].originalMin[0]
    _max = 1
    _min = 0

    print("Xmax =", Xmax, "Xmin =", Xmin, "max =", _max, "min =", _min)
    df.withColumn(colName="y_scaled_inversed", col=(_max * Xmin - _min * Xmax - Xmin * df[i+"_Scaled"] + Xmax * df[i+"_Scaled"])/(_max - _min)).show()
    

    print("After Scaling :")
    df.show(5)
