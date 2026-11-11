import json
import numpy as np
import os
from tqdm import tqdm
import subprocess
from glob import glob
import argparse
import time
from utils import crop_video, crop_face, write_video, crop_and_save_audio, remove_characters
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import sys

'''
Crop the untrimmed videos into multiple clips using corresponding start and end times, bounding boxes and face landmarks.

Usage: 
python crop_videos.py --video_dir /path/to/25-fps-videos --save_path /path/to/save/the/clips --json_path /path/to/json/folder

To save videos using ffmpeg, add "--use_ffmpeg True". This takes additional time but saves disk space.

To additionally save audio as separate wav files, add "--save_audio True"

To merge audio with video and save as a single mp4, add "--merge_audio True"
'''


def write_clip(metadata, vid_p, args):
    '''
    param metadata: dict containing start, end, bounding boxes, landmarks
    param vid_p: path to original untrimmed video at 25fps
    param args: main args 
    '''
    for k, clip in enumerate(metadata):
        # get the clip frames and corresponding landmarks
        video, landmarks = crop_video(vid_p, clip)
        # get the cropped sequence around the mouth using the landmarks
        crop_seq = crop_face(video, landmarks)
        save_video_path = os.path.join(args.save_path, 'videos', os.path.basename(vid_p)[:-4], f'{str(k).zfill(5)}.mp4')
        save_audio_path = save_video_path.replace('.mp4','.wav')
        # get the audio part of the clip
        if args.save_audio or args.merge_audio:
            crop_and_save_audio(vid_p, save_audio_path, clip['start'], clip['end'])
        # write clip to disk
        write_video(save_video_path, crop_seq, save_audio_path, merge_audio=args.merge_audio, use_ffmpeg=args.use_ffmpeg)
    return 



def main(args, batch_ids=None):
    savepath = args.save_path
    json_path = args.json_path
    vid_dir = args.video_dir

    # Indexar videos descargados por ID exacto.
    video_list = glob(os.path.join(vid_dir, '*.mp4'))
    videos_by_id = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in video_list
    }

    # Seleccionar los IDs de esta tanda.
    if batch_ids is None:
        video_ids = [
            os.path.splitext(os.path.basename(path))[0]
            for path in glob(os.path.join(json_path, '*.json'))
        ]
    else:
        video_ids = list(batch_ids)

    # Cargar únicamente los JSON de esta tanda.
    data = {}

    for video_id in video_ids:
        json_file = os.path.join(json_path, f'{video_id}.json')

        if not os.path.isfile(json_file):
            print(f'JSON no encontrado: {video_id}')
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            temp = json.load(f)

        data.update(temp)

    video_ids = list(data.keys())

    print(f'Videos descargados: {len(video_list)}')
    print(f'JSONs cargados: {len(data)}')

    # Comprobar que cada video tenga su original.
    valid_ids = []

    for video_id in video_ids:
        if video_id in videos_by_id:
            valid_ids.append(video_id)
        else:
            print(f'Video original no encontrado: {video_id}')

    video_ids = valid_ids

    futures = []
    writer_str = 'Ffmpeg' if args.use_ffmpeg else 'cv2.VideoWriter'

    print(f'Using {writer_str} to save the cropped clips.')

    errors = []

    with tqdm(total=len(video_ids), file=sys.stdout) as progress:
        with  ThreadPoolExecutor(max_workers=2) as executor:

            future_to_id = {}

            for video_id in video_ids:
                metadata = data[video_id]
                vid_p = videos_by_id[video_id]

                output_dir = os.path.join(
                    savepath, 'videos', video_id
                )
                os.makedirs(output_dir, exist_ok=True)

                future = executor.submit(
                    write_clip, metadata, vid_p, args
                )

                future_to_id[future] = video_id

            for future in as_completed(future_to_id):
                video_id = future_to_id[future]

                try:
                    future.result()
                except Exception as e:
                    errors.append((video_id, repr(e)))
                    print(f'\nERROR EN {video_id}: {repr(e)}')

                progress.update()

    print('Cropping videos completed.')

    # Construir metadata de los clips existentes de esta tanda.
    print('Getting the metadata.')

    labels = {}

    for video_id in tqdm(video_ids):
        metadata = data[video_id]

        for k, clip in enumerate(metadata):
            labk = clip['label']

            fi = os.path.join(
                savepath,
                'videos',
                video_id,
                f'{str(k).zfill(5)}.mp4'
            )

            if os.path.isfile(fi):
                labels[fi] = {
                    'label': remove_characters(labk, '<ar>'),
                    'length': len(clip['landmarks'])
                }

    # Actualizar metadata global sin eliminar registros anteriores.
    metadata_path = os.path.join(savepath, 'metadata.json')

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            global_labels = json.load(f)
    else:
        global_labels = {}

    global_labels.update(labels)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(
            global_labels,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f'Metadata global: {len(global_labels)} clips registrados.')
    print(f'Clips agregados/actualizados: {len(labels)}')
    print(f'Metadata saved to {metadata_path}')

    if errors:
        print(f'\nVideos con errores de procesamiento: {len(errors)}')
        for video_id, error in errors:
            print(video_id, error)

    return {
        'processed_ids': video_ids,
        'errors': errors,
        'clips_added': len(labels)
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViSpeR crop videos')
    parser.add_argument('--save_path', type=str, default='', help='Path for saving.')
    parser.add_argument('--json_path', type=str, default='', help='Json folder path')
    parser.add_argument('--video_dir', type=str, default='', help='Path to directory where original videos are stored.')
    parser.add_argument('--save_audio', type=bool, default=False, help='Whether to save audio info.')
    parser.add_argument('--merge_audio', type=bool, default=False, help='Whether to merge audio with the video when saving.')
    parser.add_argument('--use_ffmpeg', type=bool, default=False, help='Whether to use ffmpeg instead of cv2 for saving the video.')
    
    args = parser.parse_args()
    tic = time.time()
    main(args)
    print(f'Elpased total time for processing: {time.time()-tic} seconds')
