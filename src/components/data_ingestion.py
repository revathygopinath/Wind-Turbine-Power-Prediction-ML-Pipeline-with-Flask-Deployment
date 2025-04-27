import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from src.components import model_trainer
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset
            df = pd.read_csv(r'E:\ML Project\notebook\DATA\DATA SET.csv')
            logging.info('Read the dataset as dataframe')

            # Convert 'Date/Time' column with the correct format
            df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M', errors='coerce')

            # Drop rows where Date/Time is NaT after conversion
            df = df.dropna(subset=['Date/Time'])

            # Extract Date/Time features
            df['Month'] = df['Date/Time'].dt.month
            df['Day'] = df['Date/Time'].dt.day
            df['Year'] = df['Date/Time'].dt.year
            df['Hour'] = df['Date/Time'].dt.hour
            df['week'] = df['Date/Time'].dt.isocalendar().week

            # Create 'Seasons' column
            seasons_dict = {1: 'Winter', 2: 'Winter', 3: 'Winter', 4: 'Spring', 5: 'Spring', 6: 'Summer',
                            7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
            df['Seasons'] = df['Month'].map(seasons_dict)

            # Drop the original 'Date/Time' column as it's no longer needed
            df.drop(columns=['Date/Time'], axis=1, inplace=True)

            # Save the raw data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the split datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Now, the data is ready for transformation
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

 

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=model_trainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

