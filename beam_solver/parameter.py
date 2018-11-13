"""
Define hera_cat parameters
"""
class catParam(object):
    def __init__(self, name, description=None, value=None, expected_type=None, form=None):
        """catParam init"""
        self.name = name
        self.description = description
        self.value = value
        self.expected_type = expected_type
        self.form = form
        self.__doc__ = "name : {}, form : {}, expected_type : {} \n {}".format(name, form, expected_type, description)
