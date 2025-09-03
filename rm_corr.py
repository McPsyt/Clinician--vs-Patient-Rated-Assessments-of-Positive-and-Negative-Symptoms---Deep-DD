import pandas as pd
import pingouin as pg

#for individual columns
def rm_corr(x_col, y_col, df):

    return pg.rm_corr(data=df, x=x_col, y=y_col, subject='ID')

if __name__ == '__main__':
    df = pd.read_excel("Auditory .xlsx")
    xs = ["lshs","ahrs"]
    ys = ["saps", "panss"]
    for x in xs:
        for y in ys:
            print(x, y)
            print(rm_corr(x_col=x, y_col=y, df=df))
            print("######################")