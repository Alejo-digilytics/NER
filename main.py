from NER_model import NER

if __name__ == '__main__':
    model = NER(encoding="latin-1", base_model="bert-base-uncased")
    # model.training()
    text = "alejo is going to india"
    model.predict(text)


    """
    model.predict(" Contact tel 03457 60 60 60 see reverse for call times Text phone 03457 125 563"
                  "used by deaf or speech impaired customers"
                  "www.hsbc.co.uk"
                  " Your Statement The Secretary STORAGE FUTURE LIMITED unit 3"
                  "Fordwater Trading EST"
                  "Ford Road Account Summary"
                  "Chertsey , Surrey  Opening Balance  342,461.09 "
                  "KT16 8HG  Payments In 227,614.00 Payments Out  338,548.81"
                  "Closing Balance 231,526.28")
    """