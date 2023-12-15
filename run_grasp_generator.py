from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=846112071703,
        saved_model_path='trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98',
        visualize=True
    )
    generator.load_model()
    generator.run()
