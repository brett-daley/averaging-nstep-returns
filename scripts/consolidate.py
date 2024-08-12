from argparse import ArgumentParser
import os
import re
import traceback

import numpy as np


REGEX = "([0-9]+)_(.*.npy)"


def main(directory: str, clean: bool):
    files = os.listdir(directory)
    get_path = lambda f: os.path.join(directory, f)
    completed = set()

    for f in files:
        match = re.match(REGEX, f)

        if match:
            seed = match.group(1)
            new_name = match.group(2)

            # Make sure we don't repeat this file
            if new_name in completed:
                continue
            completed.add(new_name)

            batch_regex = "[0-9]+_" + new_name
            batch = tuple(filter(lambda f: re.match(batch_regex, f), files))

            try:
                # Load each file's data
                data = tuple(np.load(get_path(x)) for x in batch)

                # Stack data and save as new file
                data = np.stack(data)
                save_path = get_path(new_name)
                np.save(save_path, data)
                print(f"Consolidated {len(batch)} files in", save_path)

                # Clean up
                if clean:
                    for x in batch:
                        path = get_path(x)
                        print("  Removing", path)
                        os.remove(path)

            except:
                # Error, skip to prevent data loss
                print("FAILURE generating", new_name)
                print(traceback.format_exc())
                continue


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()
    main(args.directory, args.clean)
