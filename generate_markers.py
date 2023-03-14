
from pathlib import Path
import argparse as ap

import numpy as np
import cv2

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

class ArucoMarkerGenerator:
    def __init__(self, size: int, dic_name: str, dict: str, boarder: int)-> None:
        self.size = size
        self.dict_name = dic_name
        self.arc_dict = cv2.aruco.Dictionary_get(dict)
        self.boarder = boarder
    
    def generate(self, start: int, end: int, output: Path)-> None:
        """ Generate ArUCo tags and save them to disk.  The tags are saved as
        PNG files with the name `dict_id-<id>.png` where `dict` is the
        dictionary name and `id` is the tag ID.
        
        Parameters
        ----------
        start : int
            start index of ArUCo tag IDs
        end : int
            end index of ArUCo tag IDs
        output : Path
            path to output directory of markers
        """
        for i in range(start, end):
            marker = np.zeros((self.size, self.size), dtype=np.uint8)
            marker = cv2.aruco.drawMarker(
                dictionary=self.arc_dict,
                id=i,
                sidePixels=self.size,
                img=marker,
                borderBits=self.boarder
                )
            cv2.imwrite(str(output / f'{self.dict_name}_id-{i}.png'), marker)
            print(f'ArUCo tag {i} generated!')

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='path to output directory of markers')
    parser.add_argument('-s', '--size', type=int, default=200, help='size of each marker in pixels')
    parser.add_argument('-d', '--dict', type=str, default='DICT_4X4_50', help='type of ArUCo tag to generate')
    parser.add_argument('-i', '--start', type=int, default=0, help='start index of ArUCo tag IDs')
    parser.add_argument('-e', '--end', type=int, default=50, help='end index of ArUCo tag IDs')
    parser.add_argument('-b','--boarder', type=int, default=1, help='width of the marker boarder in bits')
    args = vars(parser.parse_args())
    
    aruco = ArucoMarkerGenerator(
        size=args['size'],
        dic_name = args['dict'],
        dict=ARUCO_DICT[args['dict']],
        boarder=args['boarder']
        )

    aruco.generate(args['start'], args['end'], Path(args['output']))
