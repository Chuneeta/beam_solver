class RBL(object):
    """Object to make using reds easier

    """
    def __init__(self, reds):
        self.reds = reds

    def __getitem__(self, key):
        if type(key) is tuple:
            for red in self.reds:
                if key in red:
                    return red
                else: continue

    def __iter__(self):
        for red in self.reds:
            yield red

    def get_ubl(self, bl):
        if type(bl) is tuple:
            for red in self.reds:
                if bl in red:
                    return red[0][0], red[0][1]
                else: continue
    
