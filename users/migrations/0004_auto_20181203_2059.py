# Generated by Django 2.1.3 on 2018-12-03 20:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_auto_20181129_1506'),
    ]

    operations = [
        migrations.AlterField(
            model_name='student',
            name='current_year',
            field=models.IntegerField(default=1),
        ),
        migrations.AlterField(
            model_name='student',
            name='year_of_joining',
            field=models.IntegerField(default=2000),
        ),
    ]
