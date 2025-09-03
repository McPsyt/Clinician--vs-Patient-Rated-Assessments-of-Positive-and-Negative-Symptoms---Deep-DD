import pandas as pd
import re
import pingouin as pg

#for individual columns
def rm_corr(x_col, y_col, df):

    return pg.rm_corr(data=df, x=x_col, y=y_col, subject='ID')

def add_cols(cols, df, name_of_summed_column):
    #cols = columns to add
    sum = df[cols[0]]
    for i in range(1, len(cols)):
        sum = df[cols[i]] + sum
    df[name_of_summed_column] = sum
    return df

def get_proper_col_names(questionnaire_name, cols):
    r = []
    for elm in cols:
        r.append(questionnaire_name + elm)
    return r

def add_col_for_questionnaire(df_name_of_summed_column, i, q, df_main):

    if df_name_of_summed_column[f"{q}".upper()].iloc[i] != "-":
        name_of_summed_column = df_name_of_summed_column["Summed_cols"][i]
        cols_q = df_name_of_summed_column[f"{q}".upper()].iloc[i]
        cols_q = re.sub(f"{q}_", "", cols_q).split(",")
        cols_q = get_proper_col_names(questionnaire_name=f"{q}_", cols=cols_q)
        print(cols_q, i)
        return add_cols(cols=cols_q, df=df_main, name_of_summed_column=name_of_summed_column + q)
    else:
        return df_main

def add_all_cols():
    df_name_of_summed_column = pd.read_excel("Individual Item Comparisons Breakdown.xlsx")
    df_main = pd.read_excel("Positive Symptoms 2.xlsx")

    for i in range(len(df_name_of_summed_column["Summed_cols"])):

        df_main = (add_col_for_questionnaire(df_name_of_summed_column, i, q="lshs", df_main=df_main))
        df_main = (add_col_for_questionnaire(df_name_of_summed_column, i, q="ahrs", df_main=df_main))
        df_main = (add_col_for_questionnaire(df_name_of_summed_column, i, q="pdi", df_main=df_main))
        df_main = (add_col_for_questionnaire(df_name_of_summed_column, i, q="saps", df_main=df_main))
        df_main = (add_col_for_questionnaire(df_name_of_summed_column, i, q="panss", df_main=df_main))


    return df_main


if __name__ == '__main__':
    df = add_all_cols()
    df.to_csv("Positive Symptoms with combined cols.csv", index=False)
