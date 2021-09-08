from tensorflow import keras
import torch
import numpy as np
import cv2
from path import Path

def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
              auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        interp = cv2.INTER_AREA if r <= 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def test(model, samples):
    stride = 4
    input_shape = 320
    base_res_dir = Path("partial_results")

    for file in samples:
        results_dir = base_res_dir / f"{file.name.replace('.jpg','').replace('.png','')}"
        results_dir.makedirs_p()
        file.copy(results_dir)

        # leggo immagine bgr
        img_input = cv2.imread(file)

        # resize dell'immagine come square image mantenendo l'aspect ratio e paddando con 0 dove necessario
        img_input, ratio, ds = letterbox(img_input, new_shape=input_shape, color=(0, 0, 0), auto=False, scaleup=False)

        cv2.imwrite(results_dir / f"{file.name.replace('.jpg','').replace('.png','')}_00_letterbox.jpg", img_input)

        # trasformo da bgr a rgb e converto in matrice [batch, h, w, c] con valori tra [0,1]
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims((img_input / 255.0).astype(np.float32), 0)
        # inference
        hm, sz = model(img_input, training=False)

        # riporto in array opencv like quindi uint8 con valori compresi tra [0,255]
        img_input = (img_input * 255).astype(np.uint8).squeeze()

        # trasformo le heatmap generate e la matrice delle dimensioni (sz) in torch like format per usare la maxpool2d di torch che è più comoda
        # hm (heatmap) è una matrice 80x80x2, dove nel primo canale ci sono le heatmap dei volti con mascherina e nel canale 1 quelli senza mascherina (o viceversa ma è indifferente)
        hm, sz = hm.numpy(), sz.numpy().squeeze()

        ## di seguito ci saranno varie imwrite per scrivere le immagini di esempio, chiaramente al momento della scrittura
        # le matrici float32 con valori compresi tra [0,1]  le moltiplico per 255 a uint8 altrimenti non sarebbe possibile salvarle come immagini con opencv

        cv2.imwrite(results_dir / f"{file.name.replace('.jpg','').replace('.png','')}_01_heatmap00.jpg", (hm.squeeze()[:,:,0] * 255).astype(np.uint8)) ## salvo esempio
        cv2.imwrite(results_dir / f"{file.name.replace('.jpg','').replace('.png','')}_01_heatmap01.jpg", (hm.squeeze()[:,:,1] * 255).astype(np.uint8)) ## salvo esempio

        hm = torch.from_numpy(hm.transpose(0, 3, 1, 2))
        # simple nms, eseguo un maxpool con kernel 5x5, quindi in pratica per ogni patch 5x5 tengo il massimo
        hmax = torch.nn.functional.max_pool2d(hm, kernel_size=5, padding=2, stride=1)
        cv2.imwrite(results_dir / f"{file.name.replace('.jpg', '').replace('.png', '')}_02_heatmask00.jpg",
                    (hmax.cpu().numpy().squeeze().transpose(1,2,0)[:, :, 0] * 255).astype(np.uint8))  ## salvo esempio
        cv2.imwrite(results_dir / f"{file.name.replace('.jpg', '').replace('.png', '')}_02_heatmask01.jpg",
                    (hmax.cpu().numpy().squeeze().transpose(1,2,0)[:, :, 1] * 255).astype(np.uint8))  ## salvo esempio

        keep = (hmax == hm).float()
        cv2.imwrite(results_dir / f"{file.name.replace('.jpg', '').replace('.png', '')}_03_heatkeep00.jpg",
                    (keep.cpu().numpy().squeeze().transpose(1, 2, 0)[:, :, 0] * 255).astype(np.uint8))  ## salvo esempio
        cv2.imwrite(results_dir / f"{file.name.replace('.jpg', '').replace('.png', '')}_03_heatkeep01.jpg",
                    (keep.cpu().numpy().squeeze().transpose(1, 2, 0)[:, :, 1] * 255).astype(np.uint8))  ## salvo esempio

        hm *= keep
        hm = hm.numpy().squeeze()
        cv2.imwrite(results_dir / f"{file.name.replace('.jpg', '').replace('.png', '')}_04_heatmap200.jpg",
                    (hm.transpose(1, 2, 0)[:, :, 0] * 255).astype(np.uint8))  ## salvo esempio
        cv2.imwrite(results_dir / f"{file.name.replace('.jpg', '').replace('.png', '')}_04_heatmap201.jpg",
                    (hm.transpose(1, 2, 0)[:, :, 1] * 255).astype(np.uint8))  ## salvo esempio

        for chi in range(0, 2):
            ind = np.argpartition(hm[chi].squeeze().flatten(), -input_shape // stride)[-input_shape//stride:]

            for v in ind:
                row = v % hm[chi].shape[0]
                col = v // hm[chi].shape[0]

                if hm[chi][col, row] <= 0.1:
                    continue
                szi = int(sz[col, row] * input_shape)

                row *= stride
                col *= stride
                szi = szi // 2
                x1, y1, x2, y2 = row - szi, col - szi, row + szi, col + szi
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 >= input_shape:
                    x2 = input_shape - 1
                if y2 >= input_shape:
                    y2 = input_shape - 1

                cv2.rectangle(img_input, (x1, y1), (x2, y2), (255 * (1 - chi), 0, 255 * chi), 2)

        cv2.imwrite(results_dir / f"{file.name.replace('.jpg','').replace('.png','')}_05_final.jpg", img_input)
        # cv2.waitKey()
        print(f"tested img {file}")


def main():
    model = keras.models.load_model("weights/best", compile=False)
    samples = Path("samples").files("*.png")
    test(model, samples)


if __name__ == '__main__':
    main()