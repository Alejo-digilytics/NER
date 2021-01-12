from NER_model import NER

if __name__ == '__main__':
    model = NER(encoding="utf-8", base_model="bert-base-uncased",
                pos_dropout=0.3, tag_dropout=0.3,
                pos_dropout_2=0.3, tag_dropout_2=0.3,
                architecture="complex", middle_layer=100)
    model.training()
    # text = "Ramakant is going to india"
    # model.predict(text)


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
                  
    model.predict("Contact tel 03457 60 60 60 see reverse for call times Text phone 03457 125 563 used by deaf or "
                  speech impaired customers www.hsbc.co.uk Your Statement The Secretary STORAGE FUTURE LIMITED unit "
                  3Fordwater Trading EST Ford Road Account Summary Chertsey , Surrey Opening Balance 342,461.09 KT16 "
                  8HGPayments In 227,614.00Payments Out 338,548.81Closing Balance 231,526.28.International "
                  Bank Account Number GB66HBUK4026122205007230 March to 29 April 2020 Branch Identifier CodeHBUKGB4111 "
                  G Account Name Sortcode Account Number Sheet Number STORAGE FUTURE LIMITED 40 - 26 - 12 22050072 "
                  426Your Business Current Account details Date Payment type and details Paid out Paid in Balance 29 "
                  Mar 20 BALANCE BROUGHT FORWARD 342,461.09")
    """