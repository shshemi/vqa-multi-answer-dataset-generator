import PIL.Image
import numpy as np
import json
import os
import random


class Color:

    def __init__(self, name, hex_color):
        rgb_hex_color = hex_color.replace("#", "")
        self.__name = name
        self.__rgba = tuple(int(rgb_hex_color[i:i + 2], 16) for i in (0, 2, 4)) + (255,)

    @property
    def name(self):
        return self.__name

    @property
    def rgba(self):
        return self.__rgba

    def is_close(self, color, threshold=120):
        if type(color) is list:
            return any([self.is_close(c) for c in color])
        c1 = np.array(self.rgba)
        c2 = np.array(color.rgba if type(color) is Color else color)
        dst = np.linalg.norm(c1 - c2)
        return dst < threshold

    def __eq__(self, other):
        return np.alltrue(np.array(self.rgba) == np.array(other.rgba))

    def __str__(self):
        return self.__name


class Shape:

    def __init__(self, shape_path=None):
        self.__name = os.path.splitext(os.path.basename(shape_path))[0] if shape_path is not None else None
        self.__image = PIL.Image.open(shape_path) if shape_path is not None else None
        self.__color = None
        self.__rotation = 0

    @property
    def name(self):
        if self.color is None:
            return self.__name
        return f"{self.color.name}_{self.__name}"

    @property
    def image(self):
        return self.__image

    @property
    def color(self):
        return self.__color

    @property
    def rotation(self):
        return self.__rotation

    @property
    def size(self):
        return self.__image.size

    def new(self, color, rotation=0):
        data = np.array(self.__image, dtype=np.float32)
        data *= (np.array(color.rgba, dtype=np.float32) / 255)
        new_image = PIL.Image.fromarray(data.astype(np.uint8))
        if rotation != 0:
            new_image.rotate(rotation)
        new_shape = Shape()
        new_shape.__name = self.__name
        new_shape.__image = new_image
        new_shape.__color = color
        new_shape.__rotation = rotation
        return new_shape

    def __eq__(self, other):
        if self.__name != other.__name:
            return False
        if self.__color is not None and self.__color != other.__color:
            return False
        return True

    def __str__(self):
        return self.__name


class Background:

    def __init__(self, colors, size, background_type="solid"):
        if type(colors) is not list:
            colors = [colors]
        self.__size = size
        self.__background_type_table = {
            "solid": self.__solid,
            "dual_color_vertical": self.__dual_color_vertical,
            "dual_color_horizontal": self.__dual_color_horizontal,
        }
        if background_type == "random":
            background_type = random.choice([*self.__background_type_table])
        self.__type = background_type
        if background_type == "solid":
            while len(colors) > 1:
                colors.pop()
        self.__colors = colors

    @property
    def name(self):
        str_color = "_".join([c.name for c in self.colors])
        return f"{str_color}_{self.__type}"

    @property
    def colors(self):
        return self.__colors

    @property
    def size(self):
        return self.__size

    def new(self):
        return self.__background_type_table[self.__type]()

    def __solid(self):
        bg = PIL.Image.new("RGBA", self.size, self.colors[0].rgba)
        return bg

    def __dual_color_vertical(self):
        bg = PIL.Image.new("RGBA", self.size, self.colors[0].rgba)
        bg.paste(PIL.Image.new("RGBA", (self.size[0] // 2, self.size[1]), self.colors[1].rgba),
                 (self.__size[0] // 2, 0))
        return bg

    def __dual_color_horizontal(self):
        bg = PIL.Image.new("RGBA", self.__size, self.colors[0].rgba)
        bg.paste(PIL.Image.new("RGBA", (self.size[0], self.size[1] // 2), self.colors[1].rgba),
                 (0, self.size[1] // 2))
        return bg

    def __treble_color_vertical(self):
        pass

    def __treble_color_horizontal(self):
        pass

    def __four_color_horizontal(self):
        pass

    def __four_color_vertical(self):
        pass

    def __four_color_grid(self):
        pass


class Image:

    def __init__(self, background, suffix=None):
        self.__background = background
        self.__shapes = []
        self.__image = None
        self.__suffix = suffix

    def add_shape(self, shape, position="random"):
        if position == "random":
            bsize = np.array(self.background.size)
            fsize = np.array(shape.size)
            position = [int(i) for i in np.random.uniform([0, 0], bsize - fsize).astype(np.uint)]
        elif position == "center":
            bsize = np.array(self.background.size)
            fsize = np.array(shape.size)
            position = (bsize - fsize) // 2

        self.__shapes.append((shape, tuple(position)))

    def shape_background_rgba(self, shape):
        pos = None
        for s, p in self.__shapes:
            if s == shape:
                pos = p
        return self.image.getpixel(pos)

    @property
    def background(self):
        return self.__background

    @property
    def image(self):
        if self.__image is not None:
            return self.__image
        image = self.__background.new()
        for shape, position in self.__shapes:
            image.paste(shape.image, position, shape.image)
        self.__image = image
        return image

    @property
    def shapes(self):
        return [shape for shape, _ in self.__shapes]

    @property
    def name(self):
        import hashlib
        name = self.background.name
        if len(self.shapes) > 0:
            str_shapes = "_".join([s.name for s in self.shapes])
            if self.__suffix is None:
                name = f"{str_shapes}_on_{self.background.name}"
            else:
                name = f"{str_shapes}_on_{self.background.name}_{self.__suffix}"
        return hashlib.md5(name.encode()).hexdigest()

    def save(self, dir):
        path = os.path.join(dir, self.name) + ".png"
        self.image.save(path)


class ColorPool:
    __default_instance = None

    def __init__(self, color_file):
        self.__colors = []
        with open(color_file, "r") as file:
            for jcolor in json.load(file):
                self.__colors.append(Color(jcolor["name"], jcolor["color"]))

    def random(self, exclude=[], count=1):
        if count == 1:
            return random.choice(self.colors(exclude))
        return random.sample(self.colors(exclude), count)

    def colors(self, exclude=[]):
        colors = []
        for color in self.__colors:
            if color not in exclude:
                colors.append(color)
        return colors

    def colors_len(self):
        return len(self.colors())

    @staticmethod
    def default():
        if ColorPool.__default_instance is None:
            ColorPool.__default_instance = ColorPool("colors.json")
        return ColorPool.__default_instance


class ShapePool:
    __default_instance = None

    def __init__(self, shapes_dir):
        self.__shapes = []
        for file in os.listdir(shapes_dir):
            if os.path.splitext(file)[1] == ".png":
                shape_path = os.path.join(shapes_dir, file)
                self.__shapes.append(Shape(shape_path))

    def random(self, exclude=[]):
        return random.choice(self.shapes(exclude))

    def shapes(self, exclude=[]):
        shapes = []
        for shape in self.__shapes:
            if shape not in exclude:
                shapes.append(shape)
        return shapes

    def shapes_len(self):
        return len(self.shapes())

    @staticmethod
    def default():
        if ShapePool.__default_instance is None:
            ShapePool.__default_instance = ShapePool("shapes")
        return ShapePool.__default_instance


def pool_iteration(*args):
    import itertools
    pool_map = {
        "colors": ColorPool.default().colors(),
        "shapes": ShapePool.default().shapes()
    }
    new_args = []
    for arg in args:
        if type(arg) is str:
            new_args.append(pool_map[arg])
        elif type(arg) is int:
            new_args.append(range(arg))
        else:
            new_args.append(arg)
    if len(new_args) > 1:
        return itertools.product(*new_args)
    return iter(new_args[0])


class BaseQuestion:

    def __init__(self, image, color_pool=ColorPool.default(), shape_pool=ShapePool.default()):
        self.image = image
        self.color_pool = color_pool
        self.shape_pool = shape_pool
        self.__questions = []
        self.create_questions()

    @property
    def questions(self):
        return self.__questions

    @property
    def type(self):
        return self.get_type()

    def create_questions(self):
        pass

    def get_type(self):
        return ""

    def add_question(self, question, answer, default_answer=None):
        if type(answer) is not list:
            answer = [answer]
        if len(answer) != 1 and default_answer is not None:
            answer += [default_answer]
        answer = [str(obj) for obj in answer]
        answer = list(set(answer))
        self.__questions.append({
            "question": question,
            "answer": answer,
            "image": self.image.name,
            "type": self.type
        })


class BackgroundColorQuestion(BaseQuestion):

    def get_type(self):
        return "color"

    def create_questions(self):
        self.add_question("what color is the background", self.image.background.colors)


class ShapeTypeQuestion(BaseQuestion):

    def get_type(self):
        return "object"

    def create_questions(self):
        self.add_question("what is the shape", self.image.shapes, "unknown_shape")


class ColorShapeTypeQuestion(BaseQuestion):

    def get_type(self):
        return "color object"

    def create_questions(self):
        for c in pool_iteration("colors"):
            answers = []
            for shape in self.image.shapes:
                if shape.color == c:
                    answers.append(shape)
            self.add_question(f"what is the {c} shape", answers, "unknown_shape")


class ShapeColorQuestion(BaseQuestion):

    def get_type(self):
        return "color detection"

    def create_questions(self):
        self.add_question("what color is the shape",
                          [s.color for s in self.image.shapes],
                          "unknown_color")


class ShapeTypeColorQuestion(BaseQuestion):

    def get_type(self):
        return "object color detection"

    def create_questions(self):
        for s1 in pool_iteration("shapes"):
            answers = []
            for s2 in self.image.shapes:
                if s1 == s2:
                    answers.append(s2.color)
            self.add_question(f"what color is the {s1.name}",
                              answers,
                              "unknown_color")


class CloseShapeColorQuestion(BaseQuestion):

    def get_type(self):
        return "color detection"

    def create_questions(self):
        self.add_question("is background color close to shape color",
                          ["yes" if s.color.is_close(self.image.shape_background_rgba(s)) else "no" for s in
                           self.image.shapes],
                          "no")


class ShapeExistenceQuestion(BaseQuestion):

    def get_type(self):
        return "detection"

    def create_questions(self):
        self.add_question("is there a shape",
                          ["yes" if len(self.image.shapes) > 0 else "no"],
                          "no")


class ColorExistenceQuestion(BaseQuestion):

    def get_type(self):
        return "color detection"

    def create_questions(self):
        for c in pool_iteration("colors"):
            self.add_question(f"is there a {c.name} shape",
                              ["yes" if c == s.color else "no" for s in self.image.shapes],
                              "no")


class ShapeTypeExistenceQuestion(BaseQuestion):

    def get_type(self):
        return "object detection"

    def create_questions(self):
        for s1 in pool_iteration("shapes"):
            self.add_question(f"is there a {s1.name}",
                              ["yes" if s1 == s2 else "no" for s2 in self.image.shapes],
                              "no")


class QuestionsGroup:

    def __init__(self, question_types=None):
        if question_types is None:
            self.question_types = [
                BackgroundColorQuestion,
                ShapeTypeQuestion,
                ColorShapeTypeQuestion,
                ShapeColorQuestion,
                ShapeTypeColorQuestion,
            ]
        else:
            self.question_types = question_types

    def generate_all(self, image):
        all_questions = []
        for qt in self.question_types:
            all_questions.extend(qt(image).questions)
        return all_questions

    def generate_random(self, image, type_count=1):
        questions = []
        for qt in random.sample(self.question_types, type_count):
            questions.extend(qt(image).questions)
        return questions


def show_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def generate_train_validation(output_dir, image_repeats=1, balance_approach="group", validation_split_ratio=0.01):
    # generate data
    print()
    all_data = []
    all_space = pool_iteration("colors", "shapes", "colors", image_repeats)
    total_iterations = ColorPool.default().colors_len() * \
                       ShapePool.default().shapes_len() * \
                       ColorPool.default().colors_len() * \
                       image_repeats
    for i, (bg_color, shape, shape_color, image_index) in enumerate(all_space):
        if bg_color == shape_color:
            continue

        # generate invalid image and question
        bg = Background(bg_color, (50, 50))
        image = Image(bg)
        image.save(os.path.join(output_dir, "images"))
        questions = QuestionsGroup().generate_all(image)
        all_data.extend(questions)
        bg = Background(bg_color, (50, 50))

        # generate valid image and question
        image = Image(bg, image_index + 1) if image_repeats > 1 else Image(bg)
        image.add_shape(shape.new(shape_color))
        image.save(os.path.join(output_dir, "images"))
        questions = QuestionsGroup().generate_all(image)
        all_data.extend(questions)
        show_progress(i, total_iterations, prefix="Train Generation")

    print()
    print("Total generated questions:", len(all_data))

    # balance data
    random.shuffle(all_data)
    data_table = {}
    for d in all_data:
        key = d["answer"][0]
        if key not in data_table:
            data_table[key] = []
        data_table[key].append(d)
    answer_count = {k: len(v) for k, v in data_table.items()}

    print("Number of each answer:")
    for answer, count in answer_count.items():
        print(answer, count)

    balanced_data = []

    if balance_approach == "categorical":
        data_groups = {
            "colors": [c.name for c in pool_iteration("colors")] + ["unknown_color"],
            "shapes": [s.name for s in pool_iteration("shapes")] + ["unknown_shape"]
        }

        for gname, glist in data_groups.items():
            print(gname, glist)

        print("Group minimums:")
        for gname, glist in data_groups.items():
            group_min = min([answer_count[a] if a in answer_count else 0 for a in glist])
            print(gname, group_min)
            if group_min == 0:
                continue
            for a in glist:
                balanced_data.extend(data_table[a][:group_min])

    elif balance_approach == "blindfold":
        answers_min_count = min([v for _, v in answer_count.items()])
        for answer, qlist in data_table.items():
            balanced_data.extend(qlist[:answers_min_count])

    elif balance_approach == "none":
        balanced_data = all_data

    else:
        raise Exception(f"{balance_approach} is not a valid data balance approach")

    print("Total number of questions of balancing", len(balanced_data))
    data_table = {}
    for d in balanced_data:
        key = d["answer"][0]
        if key not in data_table:
            data_table[key] = []
        data_table[key].append(d)
    answer_count = [(k, len(v)) for k, v in data_table.items()]

    print("Number of each answer after balancing:")
    for answer, count in answer_count:
        print(answer, count)

    # spliting to validation and test
    from sklearn.model_selection import train_test_split
    data_train, data_validation = train_test_split(balanced_data, test_size=validation_split_ratio)
    print("Total number of train data", len(data_train))
    print("Total number of validation data", len(data_validation))

    with open(os.path.join(output_dir, "questions_train.json"), "w") as file:
        json.dump(data_train, file)
        print("Train data file:", file.name)

    with open(os.path.join(output_dir, "questions_validation.json"), "w") as file:
        json.dump(data_validation, file)
        print("Validation data file:", file.name)


def generate_test(output_dir, image_count, min_shapes=7, max_shapes=15):
    data = []
    for i in range(image_count):
        bg = Background(ColorPool.default().random(count=2), (500, 500), "random")
        image = Image(bg, i)
        for _ in range(np.random.randint(min_shapes, max_shapes + 1)):
            image.add_shape(ShapePool.default().random().new(color=ColorPool.default().random(bg.colors)))
        image.save(os.path.join(output_dir, "images_test"))
        questions = QuestionsGroup().generate_all(image)
        show_progress(i, image_count, prefix="Test Generation")
        data.extend(questions)

    print()
    with open(os.path.join(output_dir, "questions_test.json"), "w") as file:
        json.dump(data, file)
        print("Test data file:", file.name)


if __name__ == '__main__':
    import shutil
    import argparse
    import sys

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-o", "--output-directory-name", default="synthetic_vqa", help="Output directory name")
    argument_parser.add_argument("--image-repeat", type=int, default=1)
    argument_parser.add_argument("--balance-approach", choices=["categorical", "blindfold", "none"], default="categorical")
    argument_parser.add_argument("--validation-split", type=float, default=0.01)
    argument_parser.add_argument("--test-image-count", type=int, default=1000)
    argument_parser.add_argument("--test-min-shapes-count", type=int, default=1)
    argument_parser.add_argument("--test-max-shapes-count", type=int, default=15)
    args = argument_parser.parse_args()
    print(args)

    output_dir_name = args.output_directory_name
    output_dir = os.path.join("outputs", output_dir_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, "images"))
    os.mkdir(os.path.join(output_dir, "images_test"))
    generate_train_validation(output_dir, args.image_repeat, args.balance_approach, args.validation_split)
    generate_test(output_dir, args.test_image_count, args.test_min_shapes_count, args.test_max_shapes_count)
