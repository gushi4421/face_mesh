from pathlib import Path
import cv2 as cv
import sys
import time
import argparse

from .face_detect import FaceMeshDetector
from .log import logger
from .tools.load_config import load_config
from .tools.open import open_camera, open_source


def main(args):
    if args.source:
        capture = open_source(args.source)
    else:
        capture = open_camera()

    capture.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    print("视频源 FPS: ", capture.get(cv.CAP_PROP_FPS))

    # 记录时间，计算帧率
    prev_time, current_time = 0, 0
    face_mesh_detector = FaceMeshDetector(
        model_path=args.model_path, max_faces=args.maxfaces
    )
    with FaceMeshDetector(model_path=args.model_path, max_faces=args.maxfaces) as detector:
        while capture.isOpened():
            status, frame = capture.read()
            if not status:
                logger.error("无法读取视频帧")
                break
            # 翻转图像
            frame = cv.flip(frame, 1)

            processed_frame, skeleton_img, landmarks = detector.find_face_mesh(
                frame=frame, draw=True
            )

            if args.print_landmarks and landmarks:
                logger.info(f"检测到人脸网格: {landmarks}")
                for i, landmark in enumerate(landmarks):
                    print(f"人脸 {i} 的关键点: {landmark}")

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            text = f"FPS: {int(fps)}"

            cv.putText(
                img=processed_frame,
                text=text,
                org=(10, 30),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
            )

            dst = detector.img_combine(processed_frame, skeleton_img)
            cv.imshow("Face Mesh", dst)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    config = load_config()
    config = config.get("face-mesh")
    argparser = argparse.ArgumentParser(description="面部检测")
    argparser.add_argument("--source", type=str, default=None, help="视频源路径")
    argparser.add_argument(
        "--width", type=int, default=config.get("width"), help="视频宽度"
    )
    argparser.add_argument(
        "--height", type=int, default=config.get("height"), help="视频高度"
    )
    argparser.add_argument(
        "--print_landmarks",
        action="store_true",
        default=config.get("print_landmarks", False),
        help="是否打印人脸关键点",
    )
    argparser.add_argument(
        "--model_path",
        type=str,
        default=config.get("model_path"),
        help="加载 task 路径",
    )
    argparser.add_argument(
        "--maxfaces", type=int, default=config.get("maxfaces"), help="最大识别脸数"
    )
    args = argparser.parse_args()
    main(args)
