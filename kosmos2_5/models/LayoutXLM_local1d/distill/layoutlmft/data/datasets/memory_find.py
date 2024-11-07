from transformers import XLMRobertaTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from memory_profiler import profile
from pympler.tracker import SummaryTracker

from layoutlmft.data.datasets.mpdfs_full import mpdfsFullDataset
from layoutlmft.data.data_collator import DataCollatorForMaskedVisualLanguageModeling
import gc


class args():
    def __init__(self):
        self.data_dir = "/mnt/localdata/users/tengchaolv/prj/layoutxlm/dataset"

# @profile(precision=8)
def test(data):
    # tracker = SummaryTracker()
    for i, batch in enumerate(tqdm(data)):
        # break
        if i % 100 == 0: break
        # continue
    # gc.collect()
    # tracker.print_diff()


def dump_garbage():
    """
    show us what's the garbage about
    """
    # force collection
    print("\nGARBAGE:")
    # gc.collect()
    _unreachable = gc.collect()
    print('unreachable object num:%d' % _unreachable)
    # print("\nGARBAGE OBJECTS:")
    # for x in gc.garbage:
    #     s = str(x)
    #     if s.startswith("gc: collectable"):
    #         continue
    #     if len(s) > 80: s = s[:80]
    #     print(type(x),"\n  ", s)

if __name__ == '__main__':
    gc.enable()
    # gc.set_debug(gc.DEBUG_LEAK)


    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

    train_dataset = mpdfsFullDataset(args(), tokenizer)

    data_collator = DataCollatorForMaskedVisualLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )

    data = DataLoader(
        train_dataset,
        batch_size=16,
        collate_fn=data_collator,
        num_workers=8,
    )

    test(data)

    dump_garbage()






