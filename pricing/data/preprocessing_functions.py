def select_where(column, equals_to):
    def foo_hand(df):
        df = df.loc[df[column] == equals_to]
        return df
    return foo_hand


def select_where_lessthan(column, less_than):
    def foo_hand(df):
        df = df.loc[df[column] < less_than]
        return df
    return foo_hand


def select_where_greaterthan(column, greater_than):
    def foo_hand(df):
        df = df.loc[df[column] > greater_than]
        return df
    return foo_hand

