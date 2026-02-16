import harp
import argparse
import pandas as pd
from pathlib import Path

class HarpReader:
    def __init__(self, harp_dir) -> harp.reader.DeviceReader:
        """ Reads harp data

        Parameters
        ----------
        harp_dir : str
            Directory of harp binaries
        """
        self.harp_dir = harp_dir
        self.reader = self._read_harp()
    
    def _read_harp(self):
        """ Returns harp reader object

        Returns
        -------
        harp.reader.DeviceReader
        """

        return harp.create_reader(self.harp_dir, keep_type=True)

    @property
    def get_encoder(self) -> pd.core.frame.DataFrame:
        """ Returns encoder data

        Returns
        -------
        pandas dataframe
        """

        return self.reader.AnalogData.read()["Encoder"].to_frame()
    
    @property
    def get_photodiode(self) -> pd.core.frame.DataFrame:
        """ Returns photodiode data

        Returns
        -------
        pandas dataframe
        """

        return self.reader.AnalogData.read()["AnalogInput0"].to_frame()

    @property
    def get_licks(self) -> pd.core.frame.DataFrame:
        """ Returns lick data

        Returns
        -------
        pandas dataframe
        """

        return self.reader.DigitalInputState.read()["DIPort0"].to_frame()
    
    @property
    def get_rewards(self) -> pd.core.frame.DataFrame:
        """ Returns reward data

        Returns
        -------
        pandas dataframe
        """

        return self.reader.OutputSet.read()["SupplyPort0"].to_frame()
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read harp data")
    parser.add_argument("--dir", type=str, help="Directory of harp binaries")
    parser.add_argument("--save",default=False,action='store_true',
                        help="Whether to save dataframes")
    args = parser.parse_args()

    harp_dir = Path(args.dir)

    reader = HarpReader(harp_dir)
    
    
    if args.save:
        encoder_df = reader.get_encoder
        photodiode_df = reader.get_photodiode
        licks_df = reader.get_licks
        rewards_df = reader.get_rewards
        
        save_dir = Path(harp_dir) / 'extracted_files' 
        save_dir.mkdir()
        
        
        f = save_dir / 'encoder.pkl'
        encoder_df.to_pickle(f)
        
        f = save_dir / 'photodiode.pkl'
        photodiode_df.to_pickle(f)
        
        f = save_dir / 'licks.pkl'
        licks_df.to_pickle(f)
        
        f = save_dir / 'rewards.pkl'
        rewards_df.to_pickle(f)
    
    
    else:

        
        
        print(f"Encoder data:\n{reader.get_encoder}\n\n")
        print(f"Photodiode data:\n{reader.get_photodiode}\n\n")
        print(f"Lick data:\n{reader.get_licks}\n\n")
        print(f"Reward data:\n{reader.get_rewards}\n\n")