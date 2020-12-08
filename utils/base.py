from transformers import BertTokenizer


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def have_chinese(string):
    for s in string:
        if is_chinese(s):
            return True
    return False


class MyBertTokenizer(BertTokenizer):
    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=['[unused1]'],
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            **kwargs
    ):
        super(MyBertTokenizer, self).__init__(
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=['[unused1]', '[unused2]', '[unused3]', '[unused4]','[unused5]','[unused6]'],
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            **kwargs)

    def special_decode(self, tokens):
        tokens = self.convert_ids_to_tokens(tokens)
        tokens = " ".join(tokens).replace("##", "").strip()
        tokens = tokens.split(' ')
        for i, t in enumerate(tokens):
            if t == '[unused1]':
                tokens[i] = ' '
            elif t == '[unused2]':
                tokens[i] = '\xa0'
            elif t == '[unused3]':
                tokens[i] = '\u3000'
            elif t == '[unused4]':
                tokens[i] = '“'
            elif t == '[unused5]':
                tokens[i] = '”'
            elif t == '[unused6]':
                tokens[i] = '  '
        return tokens

    def encode(
            self,
            text,
            text_pair=None,
            add_special_tokens=True,
            max_length=None,
            stride=0,
            truncation_strategy="longest_first",
            pad_to_max_length=False,
            return_tensors=None,
            **kwargs
    ):
        def _find_black(string):
            new_string = ''
            index = 0
            while index < len(string):
                s = string[index]
                if s == ' ' and 1 < index < len(string) - 1 and is_chinese(string[index - 1]) and is_chinese(
                        string[index + 1]):
                    new_string += ' [unused1] '
                elif s == '\xa0':
                    new_string += ' [unused2] '
                elif s == '\u3000':
                    new_string += ' [unused3] '
                elif s == '“':
                    new_string += ' [unused4] '
                elif s == '”':
                    new_string += ' [unused5] '
                elif string[index:index+2] == '  ':
                    new_string += ' [unused6] '
                    index += 1
                else:
                    new_string += s
                index += 1
            return new_string

        text = _find_black(text)
        return super(MyBertTokenizer, self).encode(text,
                                                   text_pair=None,
                                                   add_special_tokens=add_special_tokens,
                                                   max_length=max_length,
                                                   stride=stride,
                                                   truncation_strategy=truncation_strategy,
                                                   pad_to_max_length=pad_to_max_length,
                                                   return_tensors=return_tensors,
                                                   **kwargs)


