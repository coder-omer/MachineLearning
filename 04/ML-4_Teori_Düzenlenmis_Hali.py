#%% ML-Session-4
######## REGULARIZATION - OVERVIEW
# Kullanıldığı durumlara bakalım
    # Multicollinearity: Featurelar arasında çok yüksek ilişki varsa bu yüksek ilişkiye denir.
        # Neden istemeyen durumdur peki? Estimator ımızın unstable olmasına sebep olur. Çünkü bu
        # .. feature lar arasındaki yüksek korelasyon birbirini etkiler ve estimator unstable olur. 
        # .. Korelasyonun -(eksi) olması bir sorun olmaz ama -(eksi) yönde yüksek ilişki varsa bu da sıkıntı
        # Bunu aşmak için regularization bize fayda sağlıyor. Bunu ilerleyen kısımlarda göreceğiz

    # OLS(Ordinary Least Square)
        # OLS ile biz best line ı buluyorduk. Buradaki regularization ı yaparken yapacağımız işlem
        # .. bir penalty(ceza) ekliyoruz. Overfitting durumunda penalty ekleyerek overfitting den kurtarmaya çalışıyoruz
        # Yani burada da ceza ekleyerek regularization yapıyoruz

######## Ridge ve Lasso ile regularization 
    # Mülakat sorusu: Ridge ve lasso nedir?     # .. gibi sorular gelebilir(Altta bahsedilecek).
    # Mülakat sorusu: Overfitting i nasıl çözersin? (Cevap: ridge ile, lasso ile)
    # Ridge de residual a penalty ekliyor. Bu da formül -->  "lambda * Eğimin(Bj) kare toplamı"
    # Lasso de residual a penalty ekliyor Bu da  formül --> "lamda * Eğimin mutlak değer toplamı"
    # l0: scaler bir sayı
    # Lasso Regression(L1): mutlak değer
    # Ridge Regression(L2): square
    
######## RIDGE
    # 2 nokta olduğunu düşünelim. Modele bunu öğren dedik. İki noktadan geçecek şekilde bir fit çizgisi çizdi diyelim.
    # Test e gidince model öğrenememiş olacak test hatası yüksek olacak
    # Biz o fit çizgisinin bir nevi yönünü değiştiriyoruz ve train de bir hatam oluşacak ama(bunu göze alıyoruz)
    # .. test durumuna gidince de hatam azalmış oluyor. Bu da bizim için overfitting e göre daha ideal bir durum.
    # Bunun ayarlamasını formüldeki "lambdayı"(parametreyi) değiştirerek ayarlayacağız
    # Özetle bir ifade(penalty) ekleyerek çizgiyi oynatmış olup test hatasını azaltıyoruz. Ridge bu şekilde çalışır
    # Normalde hata bizden kaynaklanıyor(Penalty eklediğimiz için).
    # Orion Hoca: regresyon katsayısı = coef = slope .. bunu değiştiriyoruz
    # .. Kafamıza göre bir şey eklemiyoruz. Burada slope un bilgisini kullanıyoruz.
    # Sonuç olarak, lambdanın değişik değerleri için eğim değişiyor. En minimum hatayı veren lambdayı belirleyeceğiz
    # Lambdanın nasıl belirlendiğini göreceğiz sonra
    # NOT:Ridge ve Lasso aynı işlemi yapıyor sadece biri formülde kare alıyor biri mutlak değer

### Ridge avantaj ve dezavantajları
    # Avantajları: 
        # 1.Az veri olduğunda kullanışlı 
        # 2.Multicollinearity olduğunda kullanışlı 
        # 3.Hatam oluşuyor(train de) penalty eklediğimizden ama daha düşük varyans ve düşük hata metrikleri olmasını sağlıyor sonuçta
    # Dezavantajları: 
        # 1.Feature selection için uygun değil. Featureların katsayılarını ya çok yüksek verir
        # .. ya da 0 a yaklaşır katsayılar. Çünkü formülde karesini aldığı için. Sonuç olarak feature selection yapmakta zorlanırız
    # NOT: Lasso da bazı featureları siler. Lasso da feature selection yapabilirim

######### LASSO
    # Ceza ekleme mantığı ve buna göre regularization olma mantığı Lassoda da var
    # Bu yöntemin en büyük avantajı, çok iyi tahminler yapabilmesidir.Çünkü katsayıların sıfır olması veya sıfıra
    # .. doğru daralması varyansı azaltabilmektedir
    # Ridge de katsayı 0 a hiç bir zaman gitmiyor
    # Katsayılar bize hangi feature un daha önemli olduğunu gösteriyor(Scaling yaptıktan sonra)
    # Lasso da 50 feature ı 5 feature a indirebiliyor. Ridge de 50 feature 50 olarak kalıyor
    
### Lasso avantaj ve dezavantajları
    # Avantajları:
        # 1.Feature selection için kullanışlı
        # 2.Hatam oluşuyor(train de) penalty eklediğimizden ama daha düşük varyans ve düşük hata metrikleri olmasını sağlıyor sonuçta
    # Dezavantajları :
        # 1.Feature larımın çok yüksek korelasyonu varsa birini tutup diğerlerini atabiliyor lasso

### NOT:
    # Ridge de featureları atmadığı için yüksek değerleri "featureları gruplayabilirsiniz"
    # Lasso da "featureları attığı" için kalan featurelardan seçilebilir
    # Orion hoca: feature selection yapmak istiyorsanız lassoyu kullanacağız

######### ELASTIC-NET
    # Ridge ve Lassonun toplamı(kombinasyonu). "fi" diye bir katsayı var.
    # Bu katsayının değerine göre hangi modelden ne kadar kullanacağını belirleyeceğiz
    # fi=1 olursa, sadece ridge i kullanacak
    # fi=0 olursa sadece lasso gibi çalışacak
    # fi=0.5 olursa iki yöntemi eşit oranda kullanmış olacak vs...
    # grid search kullanarak bu değeri belirleyeceğiz

# Regularization için ridge ve lasso dedik(Elastic nette kullanabiliriz dedik) Alttakiler yapılarak da regularization yapılır

######### FEATURE SCALING
# Data science da çok kullanılan yöntem. Kullanma sebeplerimiz;
# 1.Gradient descent
    # Biz ML de katsayıları optimize edip best line ı bulmaya çalışıyoruz.
    # Katsayıları optimize etmek için ilerde gradient descent kullanacağız
    # Gradient descent ile türev bilgisini kullanarak katsayıları optimize ediyoruz
    # Türev = Eğim/Değişim. Eğrinin ne kadar değiştiğini söylüyor bize. Eğimden teğet
    # .. geçirerek eğimi hesaplıyorduk
    # Discrete deki karşılığı farktır(çıkartma). a1-a2
    # Türeve bakarak adımlarda değişiklik yapıyor gradient descent
    # Gradient descent te türev olduğu için, örneğin auto-scoutta arabanın fiyatları
    # .. için tahmin ile gerçek değerler arasında fark çoksa gradient descent ayarlamayı uzun
    # .. yapıyor. O yüzden scaling yaparak gradient descent in hızlı çalışmasını sağlıyoruz
# 2.Algoritmamız uzaklık tabanlı(temelli) ise
    # Uzayda aynı ölçekte değilse. Bir değişkende çok büyük değişiklikler başkasında küçük değişiklikler
    # .. olacak scale o yüzden scaling gerekli
# 3.Katsayıların(Coefficient) anlamlı olması için
    # Model coefficient da scaling yapmazsak coefficientlar bizim için anlamlı olmaz
    # .. o yüzden scaling yapmadan coefficientlar hakkında yorum yapmak iyi değildir

### NOTLAR
    # Scaling yapmakta fayda var yapmazsak performance düşebilir
    # 2 türlü scaling var.
    # Standardization : Deviation=1, mean=0 olacak şekilde veriyi dönüştürüyor   --> StandardScaler # Formül:Xi-mean(mü)/Deviation(sigma)
    # Normalization   : Değerler 0-1 arasında olacak şekilde veriyi dönüştürüyor --> MinMaxScaler   # Formül:(Xi-X(mean))/(Xmax-Xmin)
# Class chatteki bir soru: label(Target/Bağımlı) degere(değişkene) scaling yapılmıyor?
    # Orion Hoca: ML de label kutsal. Mümkünse dokunmuyoruz.

######### MODEL FITTING
    # Fit i ve transform u train e yapıyoruz
    # Test e transform u uygulayabiliriz. fit i uygulamıyoruz.
    # John Hoca: Burada bunu bilsek yeterli

######### CROSS VALIDATION
    # Eğitim verisi alt kümelere ayrılır. Tek alt kümeyi eğitim için kullanıp diğer kalan kümeleri doğrulama işlemi için kullanılır.
    # .. Bu işlem çapraz bir şekilde tüm alt kümeler için tekrarlanır. Bu işleme çapraz doğrulama denir.
    # .. Bu işlem daha önceden belirlenen belli bir k sayısında yapılır. (Literatürde ten-cross validation ifadesine çok rastlarsınız.) 
    # .. Veri eşit boydaki k parçaya ayrılır ve k kez değerlendirili
    # Önceden Train_test_splitte elimizdeki datayı örneğin %30 test, %70 train olarak ayırıyor ve bir sonuç üretiyorduk. 
    # .. Peki Bu %30 u başka yerlerden seçseydik? Seçtiğimiz %30 kötü bir yerden seçildiyse ya da değerlerin dengesiz olduğu 
    # .. bir yerden seçildiyse?? Çözüm için; Bunu biz farklı farklı yerlerden böleyim ve hepsi için sonuç üretelim en son bunların
    # .. ortalamasını alalım diyoruz.
    # K-fold Cross validation : k değerinin kaç olacağını belirtiyoruz. Genelde 5 veya 10 seçilir
    # k=5 olursa 4 parçasını(yani %80) traine diğer 1 parçayı test e atarak yapar. 10 olursa; 9 a 1 şeklinde

######## LOO(Leave one out)
    # Bu bilgisayarı çok yoruyor. Büyük datada sıkıntı. Çok tercih edilmiyor
    # Datadan değerlerin 1 i hariç hepsini alıyor train ediyor - Tek değerle test ediyor
    # Sonra diğer 1 i ayırıyor test e kalanıyla train ediyor.. bu şekilde devam ediyor
    # En son üretilen sonucun ortalamasını alıyoruz
    # NOT: Bu prosedür genellikle veri setinde aşırı uç değerlerin varlığında kullanılmaktadır

######## HOLD-OUT
    # Hold-out, verisetini “eğitim” ve “test” kümesi olarak ikiye ayırma yöntemidir.
    # Özetle, train_test_split olarak yapılan şey
    # NOT:Holdout yöntemi çapraz doğrulamanın(Cross-validation) en basit çeşididir(https://veribilimcisi.com/2017/07/13/capraz-dogrulama-cross-validation-nedir/)

######## GRID SEARCH
    # Hyperparametreleri optimize etmek için grid search kullanıyoruz. # Grid search bütün kombinasyonları deniyor
    # Hyperparameters: Her modelin parametreleri vardır. Biz parametre için hangi değerin daha iyi olduğunu belirlemek için 
    # .. modelin performansını etkileyen parametrelerdir.
    # Orion Hoca: Modelde bizim ayarladığımız parametrelere hyperparametre diyoruz.
    
    