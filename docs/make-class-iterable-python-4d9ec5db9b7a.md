# 如何在 Python 中创建用户定义的可重复项

> 原文：<https://towardsdatascience.com/make-class-iterable-python-4d9ec5db9b7a>

## 在 Python 中创建迭代器和可迭代的用户定义类

![](img/d75823e57f14b4d42578978cea93c6c7.png)

[亨利&公司](https://unsplash.com/@hngstrm?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/loop?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 介绍

在我以前的一篇文章中，我讨论了 Iterable 和 Iterator 对象，以及它们是如何参与 Python 迭代协议的。

[](/python-iterables-vs-iterators-688907fd755f)  

在今天的文章中，我们将探索如何创建用户定义的迭代器，这些迭代器可以在用户定义的类中使用，以使它们成为可迭代的。换句话说，我们将展示如何创建一个用户定义的类，并实现所有需要的方法，以满足迭代协议中描述的 Python Iterables 的特征。

## Python 中的迭代器和可迭代对象

迭代器和可迭代对象是 Python 迭代协议的两个主要结构。简而言之，Python 中的 Iterable 是一个对象，您可以在其上迭代其元素，而 Iterator 是一个对象，它返回 Iterable 对象并用于在迭代过程中产生值。

换句话说，

*   Iterable 对象实现`__iter__()`方法并返回一个 Iterator 对象
*   Iterator 对象实现了`__next__()`方法，并在 iterable 对象的元素用尽时引发一个`StopIteration`。此外，迭代器对象本身是一个 Iterable，因为它也必须实现`__iter__()`方法并简单地返回自身。

有关 Python 中迭代协议、迭代器和可迭代对象的更多细节，请务必阅读引言中所附的文章，在该文章中我们将更详细地讨论这些概念。

## 创建用户定义的迭代器和可迭代对象

现在让我们假设我们想要创建一些对象来说明一所大学是如何运作的。显然，我将要使用的例子会过于简化，因为这里的最终目标是展示如何实现和使用可迭代对象和迭代器。

```
class Student: def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name    def __str__(self):
        return f'Student Name: {self.first_name} {self.last_name}'class Lecturer: def __init__(self, first_name, last_name, subject):
        self.first_name = first_name
        self.last_name = last_name
        self.subject = subject def __str__(self):
        return f'{self.subject} Lecturer: '
               f'{self.first_name} {self.last_name}'class UniversityClass:

    def __init__(self, lecturers=[], students=[]):
        self.lecturers = lecturers
        self.students = students def add_student(student):
        raise NotImplementedError def remove_student(student):
        raise NotImplementedError def add_lecturer(lecturer):
        raise NotImplementedError def remove_lecturer(lecturer):
        raise NotImplementedError
```

现在让我们也创建一些`Student`和`Lecturer`对象的实例，然后我们将使用它们来创建一个`UniversityClass`对象的实例:

```
s1 = Student('Andrew', 'Brown')
s2 = Student('Helen', 'White')
s3 = Student('George', 'Johnson')l1 = Lecturer('Maria', 'Richardson', 'Algorithms')
l2 = Lecturer('Bob', 'Johanson', 'Programming')uni_cl = UniversityClass(lecturers=[l1 ,l2], students=[s1, s2, s3])
```

现在让我们假设我们想要迭代一个`UniversityClass`对象的实例，以便访问每个成员，包括某个班级的讲师和学生。

```
for member in uni_cl:
    print(member)
```

上面的命令将失败，并显示一个错误，提示我们`UniversityClass`不是一个可迭代的对象。

```
Traceback (most recent call last):
  File "iterable_example.py", line 45, in <module>
    for member in uni_cl:
TypeError: 'UniversityClass' object is not iterable
```

为了能够迭代用户定义的`UniversityClass`类，我们需要创建一个迭代器，我们将在使`UniversityClass`成为可迭代对象时使用它。

如前所述，迭代器必须实现`__next__()`方法，当 Iterable 对象的元素用尽时抛出一个`StopIteration`，还必须实现返回自身实例的`__iter__()`方法。

首先，让我们创建一个迭代器对象，我们将在其中实现`__next__()`方法中的逻辑。注意，这里期望的行为是首先返回讲师，一旦这些大学类成员的集合用尽，我们将开始返回学生。

```
class UniversityClassIter: def __init__(self, university_class):
        self._lect = university_class.lecturers
        self._stud = university_class.students
        self._class_size = len(self._lect) + len(self._stud)
        self._current_index = 0 def __iter__(self):
        return self def __next__(self):
        if self._current_index < self._class_size:
            if self._current_index < len(self._lect):
                member = self._lect[self._current_index] 
            else:
                member = self._stud[
                    self._current_index - len(self._lect)] self._current_index += 1
            return member raise StopIteration
```

既然我们已经创建了适当的迭代器类，现在我们可以继续修改`UniversityClass`对象，使其实现`__iter__()`方法。

```
class UniversityClass:

    def __init__(self, lecturers=[], students=[]):
        self.lecturers = lecturers
        self.students = students def add_student(student):
        raise NotImplementedError

    def remove_student(student):
        raise NotImplementedError

    def add_lecturer(lecturer):
        raise NotImplementedError

    def remove_lecturer(lecturer):
        raise NotImplementedError def __iter__(self):
        return UniversityClassIter(self)
```

瞧啊。我们现在已经创建了一个用户定义 Iterable。让我们测试一下——这一次，如果我们试图迭代一个`UniversityClass`对象的实例，我们应该能够检索所创建的类的讲师和学生:

```
s1 = Student('Andrew', 'Brown')
s2 = Student('Helen', 'White')
s3 = Student('George', 'Johnson')l1 = Lecturer('Maria', 'Richardson', 'Algorithms')
l2 = Lecturer('Bob', 'Johanson', 'Programming')uni_cl = UniversityClass(lecturers=[l1 ,l2], students=[s1, s2, s3])for member in uni_cl:
   print(member)
```

现在输出将是:

```
Algorithms Lecturer: Maria Richardson
Programming Lecturer: Bob Johanson
Student Name: Andrew Brown
Student Name: Helen White
Student Name: George Johnson
```

## 完整代码

今天教程的完整代码也作为 GitHub 要点分享如下——在 Python 中创建自己的用户定义类和 iterables 时，可以随意使用它作为参考！

Python 中用户定义的可重复项的示例类—来源:作者

## 最后的想法

在今天的教程中，我们讨论了 Python 中的可迭代对象和迭代器，以及它们是如何参与迭代协议的。此外，我们展示了如何通过实现所有必需的方法来创建用户定义的 Iterable。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**你可能也会喜欢**

[](/python-iterables-vs-iterators-688907fd755f)  [](/requirements-vs-setuptools-python-ae3ee66e28af)  [](/how-to-merge-pandas-dataframes-221e49c41bec) 