
class Converter():
    def __init__(self,df):
        self.df = df
        self.categoricals = []

    def convert_to_horse(self,x):
        inp_dic = {}
        numericals = [c for c in self.df.columns if c not in self.categoricals]
        for c in self.categoricals:
            inp_dic[c] = self.df.loc[:,c]
        inp_dic["main"] = self.df.loc[:,numericals]
        return inp_dic


    def convert_to_race(self,x):
        pass
