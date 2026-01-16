import cv2  
import torch 
import time
import numpy as np
import os
from label import Label
from sam2.build_sam import build_sam2_video_predictor
import re
import shutil
class SAM_Annotator:
    def __init__(self, model_type="vit_b", model_cfg_path = None, ckpt_path=None):
        self.model_loaded = False
        self.model_loading = False
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model_cfg_path = model_cfg_path
        self.labels = []
        self.object_id_to_label_name = {}
        self.blocking_frames = []
        self.current_block = 0
        self.media_files = []
        self.propagation_blocks = {}
        self.loading_stages = [
            "Loading model weights",
            "Setup complete"
        ]
        self.current_stage = 0
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8: # 8 ~ Ampere
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = torch.device('cpu')
            print('Warning: only cpu available, SAM2 will not work properly')
        self.curr_img_idx = -1
        self.inference_state = None
        self.tracking_init = False
        self.object_ids = {}
        self.frame_names = None
    def load_model(self, status_callback):
        if self.model_loaded or self.model_loading:
            return True
        self.model_loading = True
        try:
            start_time = time.time()
            self._update_loading_stage(status_callback)
            self.predictor = build_sam2_video_predictor(self.model_cfg_path, self.ckpt_path).to(device=self.device)
            self._update_loading_stage(status_callback)
            self.model_loaded = True
            self.model_loading = False
            load_time = time.time() - start_time
            if status_callback:
                status_callback(f"SAM2 loaded successfully in {load_time:.1f}s (using {self.device})")
            return True
        except Exception as e:
            if status_callback:
                status_callback(f"Error loading SAM2: {str(e)}")
            self.model_loading = False
            return False
    def _update_loading_stage(self, status_callback):
        if self.current_stage < len(self.loading_stages):
            stage_text = self.loading_stages[self.current_stage]
            if status_callback:
                progress = f"[{self.current_stage+1}/{len(self.loading_stages)}]"
                status_callback(f"Loading SAM2 model {progress} {stage_text}")
            self.current_stage += 1
    def is_model_loaded(self):
        return self.model_loaded
    def _find_most_recent_prompt(self,label,idx):
        most_recent_prompt = -1
        for pt in label.pts[self.current_block]:
            if pt.idx <= idx:
                most_recent_prompt = np.amax([most_recent_prompt,pt.idx])
        return most_recent_prompt
    def _find_most_recent_prompt_box(self,label,idx):
        most_recent_prompt = -1
        for box in label.boxes[self.current_block]:
            if box.idx <= idx:
                most_recent_prompt = np.amax([most_recent_prompt,box.idx])
        return most_recent_prompt
    def _preprocess_label(self,label,idx,flag=0):
        pt_coords = []
        pt_labels = []
        boxes = []
        most_recent_prompt = -1
        most_recent_prompt_box = -1
        if flag == 1:
            most_recent_prompt = self._find_most_recent_prompt(label,idx)
        for pt in label.pts[self.current_block]:
            if flag == 0 and pt.idx == self.curr_img_idx:
                pt_coords.append([pt.x, pt.y])
                pt_labels.append(pt.pt_type)
            if flag == 1 and pt.idx == most_recent_prompt:
                pt_coords.append([pt.x, pt.y])
                pt_labels.append(pt.pt_type)
        if flag == 1:
            most_recent_prompt_box = self._find_most_recent_prompt_box(label,idx)
        for box in label.boxes[self.current_block]:
            if flag == 0 and box.idx == self.curr_img_idx:
                boxes.append([box.fx,box.fy,box.x,box.y])
            if flag == 1 and box.idx== most_recent_prompt_box:
                boxes.append([box.fx,box.fy,box.x,box.y])
            if len(boxes) > 0:
                boxes = [boxes[0]]
        pt_coords = np.array(pt_coords)
        pt_labels = np.array(pt_labels)
        boxes = np.array(boxes)
        return pt_coords, pt_labels, boxes, most_recent_prompt, most_recent_prompt_box
    def _get_object_id_for_label(self, label_name):
        if (label_name + str(self.current_block)) not in self.object_ids:
            self.object_ids[(label_name + str(self.current_block))] = len(self.object_ids) + 1
            self.object_id_to_label_name[self.object_ids[(label_name + str(self.current_block))]] = label_name
        return self.object_ids[(label_name + str(self.current_block))]
    def init_inference_state(self, path, status_callback=None):
        if not self.model_loaded:
            return False
        try:
            self.inference_state = self.predictor.init_state(video_path=path)
            self.tracking_init = True
            if status_callback:
                status_callback(f"SAM2 tracking initialized successfully!")
            return True
        except Exception as e:
            if status_callback:
                status_callback(f"Error initializing SAM2 tracking: {str(e)}")
            return False
    def generate_mask_for_frame(self,idx,flag=0):
        if not self.model_loaded:
            return {}
        try:
            print(f"### {self.media_files}")
            current_path = self.media_files[idx]
            img_dir = os.path.dirname(current_path)
            os.makedirs("./singular_temp_dir/",exist_ok=True)
            shutil.copy(current_path,os.path.join("./singular_temp_dir/",os.path.basename(current_path)))
            if not self.tracking_init or self.inference_state is None:
                temp_inference_state = self.predictor.init_state(video_path=img_dir)
            else:
                temp_inference_state = self.inference_state
            results = {}
            for label in self.labels:
                pt_coords, pt_labels, boxes, _, _ = self._preprocess_label(label,idx,flag)
                if pt_coords.shape[0] != 0:
                    with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                        _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=temp_inference_state,
                            frame_idx=self.curr_img_idx,
                            obj_id=self._get_object_id_for_label(label.name),
                            points=pt_coords,
                            labels=pt_labels
                        )
                if boxes.shape[0] != 0:
                    with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                        _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=temp_inference_state,
                            frame_idx=self.curr_img_idx,
                            obj_id=self._get_object_id_for_label(label.name),
                            box = boxes
                        )
            mask = (mask_logits > 0.0).cpu().numpy()
            frame_results = {}
            for j, obj_id in enumerate(obj_ids):
                frame_results[self.object_id_to_label_name[obj_id]] = mask[j]
            results[idx] = frame_results
            if not self.tracking_init or self.inference_state is None:
                self.predictor.reset_state(temp_inference_state)
            return results
        except Exception as e:
            print(f"Error generating mask: {str(e)}")
            return {}
    def propagate(self, direction, start_frame_idx, end_frame_idx = None, progress_callback = None, flag = 1):
        if not self.tracking_init:
            print("Tracking not initialized!")
            return {}
        try:
            self.predictor.reset_state(self.inference_state)
            for label in self.labels:
                pt_coords, pt_labels, boxes, most_recent_prompt, most_recent_prompt_box = self._preprocess_label(label,start_frame_idx,flag)
                if pt_coords.shape[0] != 0:
                    with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                        self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=most_recent_prompt,
                            obj_id=self._get_object_id_for_label(label.name),
                            points=pt_coords,
                            labels=pt_labels
                        )
                if boxes.shape[0] != 0:
                    with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                        self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=most_recent_prompt_box,
                            obj_id=self._get_object_id_for_label(label.name),
                            box = boxes
                        )
            results = {}
            prop_extra_frame = True
            if direction == 1:
                if end_frame_idx is None:
                    end_frame_idx = len(self.media_files)
                max_efi = 1000000
                for k,v in self.propagation_blocks[self.current_block].items():
                    if k > start_frame_idx and k < max_efi:
                        max_efi = k
                        prop_extra_frame = False
                end_frame_idx = np.amin([end_frame_idx,max_efi])
                if prop_extra_frame:
                    end_frame_idx += 1
                total_prop_frames = end_frame_idx - start_frame_idx - 1
            else:
                if end_frame_idx is None:
                    end_frame_idx = 0
                max_efi = -1000000
                for k,v in self.propagation_blocks[self.current_block].items():
                    if k < start_frame_idx and k > max_efi:
                        max_efi = k
                end_frame_idx = np.amax([end_frame_idx,max_efi+1])
                if end_frame_idx is not None:
                    total_prop_frames = start_frame_idx - end_frame_idx
                else:
                    total_prop_frames = start_frame_idx
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                for i, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(
                        self.predictor.propagate_in_video(self.inference_state,start_frame_idx=start_frame_idx,max_frame_num_to_track=total_prop_frames,reverse=(direction != 1))):
                    masks = (out_mask_logits > 0.0).cpu().numpy()
                    frame_results = {}
                    for j, obj_id in enumerate(out_obj_ids):
                        frame_results[self.object_id_to_label_name[obj_id]] = masks[j]
                    results[out_frame_idx] = frame_results
                    if progress_callback:
                        progress = (i / total_prop_frames) * 100
                        progress_callback(f"Processing data...", progress)
            torch.cuda.empty_cache()
            self.predictor.reset_state(self.inference_state)
            return results
        except Exception as e:
            print(f"Error propagating masks: {str(e)}")
            if progress_callback:
                progress_callback(f"Error propagating masks: {str(e)}", 0)
            return {}
    def propagate_to_all(self, current_frame_idx, start_frame_idx = None, end_frame_idx = None, progress_callback = None, flag = 1):
        forward_prop_results = self.propagate(1,current_frame_idx,end_frame_idx,progress_callback,flag)
        backward_prop_results = self.propagate(-1,current_frame_idx,start_frame_idx,progress_callback,flag)
        return forward_prop_results | backward_prop_results